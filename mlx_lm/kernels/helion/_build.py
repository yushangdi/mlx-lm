# Copyright © 2026 Apple Inc.

"""Compile a Helion kernel to the arguments of `mx.fast.metal_kernel`.

Helion runs the kernel once on MPS. We record its Metal launch and turn the
generated MSL into a header and a kernel body for MLX. Needs torch and
Helion with the Metal backend.

Adapted from muse-glimmer-et-helion, helion/tools/helion_mlx.py.
"""

import ast
import importlib
import importlib.metadata
import inspect
import re
import textwrap
from pathlib import Path

import helion
import numpy as np
import torch
from helion._compiler.metal import metal_jit

_C10_INCLUDE = re.compile(r'^\s*#include\s*[<"](c10/[^>"]+)[>"]\s*$', re.M)
_TORCH_INCLUDE = Path(torch.__file__).parent / "include"
_KERNEL = re.compile(r"(\[\[[^\]]*\]\]\s*)?kernel\s+void\s+(\w+)\s*\(", re.S)
_ATTR_PARAM = re.compile(r"^(uint3?|ushort3?)\s+(\w+)\s*\[\[(\w+)\]\]$")
_BUFFER_PARAM = re.compile(r"^(?:device|constant)\b.*?(\w+)\s*\[\[buffer\(\d+\)\]\]$")
_TG_DECL = re.compile(
    r"^\s*threadgroup\s+([\w:<>, ]+?)\s+(\w+)\s*((?:\[[^\]]+\])+)\s*;\s*$", re.M
)
_CONSTEXPR = re.compile(r"^\s*constexpr\b.*;\s*$", re.M)
_MUTABLE = (
    "template <typename T> inline T _mut(T p) { return p; }\n"
    "template <typename T> inline device T* _mut(const device T* p) "
    "{ return const_cast<device T*>(p); }\n"
)
_TORCH_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "int16": torch.int16,
}


def _stored(body, names):
    """The tensor parameters the body stores to (tl.store or an atomic)."""
    params, stored = set(names), set()
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "tl"
            and (node.func.attr == "store" or node.func.attr.startswith("atomic_"))
            and node.args
        ):
            stored.update(
                n.id
                for n in ast.walk(node.args[0])
                if isinstance(n, ast.Name) and n.id in params
            )
    return stored


def _capture(kernel, args):
    """Run kernel(*args) and record each Metal launch."""
    launches = []
    orig = metal_jit._MetalKernel.launch

    def record(self, grid, block_dims, launch_args, by_value):
        orig(self, grid, block_dims, launch_args, by_value)
        fn_def = next(
            n
            for n in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(self._fn))))
            if isinstance(n, ast.FunctionDef)
        )
        names = [a.arg for a in fn_def.args.args]
        stored = _stored(fn_def.body, names)
        if any("helion_phase" in ast.unparse(n) for n in fn_def.body):
            raise NotImplementedError("barrier kernels are not supported")
        rec = [
            (n, a.detach().clone() if isinstance(a, torch.Tensor) else a, n in stored)
            for n, a in zip(names, launch_args, strict=True)
        ]
        launches.append((self.msl_source, (*grid, 1, 1)[:3], tuple(block_dims), rec))

    metal_jit._MetalKernel.launch = record
    try:
        kernel(*args)
        torch.mps.synchronize()
    finally:
        metal_jit._MetalKernel.launch = orig
    return launches


def _inline_c10(text, seen):
    def repl(m):
        rel = m.group(1)
        if rel in seen:
            return ""
        seen.add(rel)
        src = (_TORCH_INCLUDE / rel).read_text().replace("#pragma once", "")
        return _inline_c10(src, seen)

    return _C10_INCLUDE.sub(repl, text)


def _outline(fn, tensors, attrs, scalars, body):
    """Move the kernel body into a header function. Threadgroup arrays must
    be in kernel scope, so they stay in the source and pass by reference."""
    decls = list(_TG_DECL.finditer(body))
    kernel_scope = [m.group(0).strip() for m in decls]
    for m in reversed(decls):
        body = body[: m.start()] + body[m.end() :]
    dims = " ".join(m.group(3) for m in decls)
    sizes = []
    for c in _CONSTEXPR.findall(body):
        m = re.search(r"\b(\w+)\s*=", c)
        if m and m.group(1) in dims:
            sizes.append(c.strip())
    pragmas = re.findall(r"^\s*#pragma [^\n]*\n", body, flags=re.M)
    for p in pragmas:
        body = body.replace(p, "", 1)
    scalars = [p.strip() for p in pragmas] + scalars
    fparams = [f"T_{t} {t}" for t in tensors]
    fparams += [f"{typ} {alias}" for typ, alias, _ in attrs]
    fparams += [f"threadgroup {m.group(1)} (&{m.group(2)}){m.group(3)}" for m in decls]
    function = (
        f"template <{', '.join(f'typename T_{t}' for t in tensors)}>\n"
        f"inline void {fn}({', '.join(fparams)}) {{\n"
        + "\n".join(scalars)
        + "\n"
        + body
        + "\n}\n"
    )
    passed = list(tensors)
    if "tensor<device " in body:
        # An MPP tensor over an input needs a mutable pointer.
        function = _MUTABLE + function
        passed = [f"_mut({t})" for t in tensors]
    call = passed + [a for _, _, a in attrs] + [m.group(2) for m in decls]
    source = "\n".join(sizes + kernel_scope + [f"{fn}({', '.join(call)});"]) + "\n"
    return function, source


def _to_spec(name, launch):
    msl, grid, block, args = launch
    m = _KERNEL.search(msl)
    header = _inline_c10(msl[: m.start()], set())
    depth, i = 1, m.end()
    while depth:
        depth += {"(": 1, ")": -1}.get(msl[i], 0)
        i += 1
    params = [p.strip() for p in msl[m.end() : i - 1].split(",\n")]
    body = msl[msl.index("{", i) + 1 : msl.rindex("}")]
    values = {n: a for n, a, _ in args}
    attrs, scalars = [], []
    for p in params:
        if am := _ATTR_PARAM.match(p):
            attrs.append(am.groups())
            continue
        var = _BUFFER_PARAM.match(p).group(1)
        if not isinstance(values[var], torch.Tensor):
            # A runtime scalar: the value becomes a constant.
            ctype = p.split()[1].rstrip("&")
            scalars.append(f"    const {ctype} {var} = {values[var]!r};")
    inputs = [(n, a) for n, a, s in args if isinstance(a, torch.Tensor) and not s]
    outputs = [(n, a) for n, a, s in args if isinstance(a, torch.Tensor) and s]
    fn = f"{name}_body"
    function, source = _outline(
        fn, [n for n, _ in inputs + outputs], attrs, scalars, body
    )
    dt = lambda t: str(t.dtype).removeprefix("torch.")
    spec = {
        "name": name,
        "source": source,
        "header": header + function,
        "input_names": [n for n, _ in inputs],
        "input_shapes": [list(a.shape) for _, a in inputs],
        "input_dtypes": [dt(a) for _, a in inputs],
        "output_names": [n for n, _ in outputs],
        "output_shapes": [list(a.shape) for _, a in outputs],
        "output_dtypes": [dt(a) for _, a in outputs],
        "grid": [g * b for g, b in zip(grid, block)],
        "threadgroup": list(block),
    }
    return spec, [a for _, a in inputs], [a for _, a in outputs]


def _example(shape, dtype, gen):
    if dtype == "int16":
        return torch.randint(-(2**15), 2**15, shape, generator=gen, dtype=torch.int16)
    return (torch.randn(shape, generator=gen) * 0.1).to(_TORCH_DTYPES[dtype])


def build(name, inputs, scalars, effort, check):
    """The spec of the Helion kernel `name` (the function of the module of
    the same name) for inputs, a list of (name, shape, dtype), and scalars.
    check(spec, inputs, expected) runs the spec in MLX and must reproduce
    Helion's outputs."""
    fn = getattr(importlib.import_module(f"{__package__}.{name}"), name)
    gen = torch.Generator().manual_seed(0)
    tensors = {n: _example(s, d, gen).to("mps") for n, s, d in inputs}
    args = [
        tensors[p] if p in tensors else scalars[p]
        for p in inspect.signature(fn).parameters
    ]
    kernel = helion.kernel(
        fn, backend="metal", static_shapes=True, autotune_effort=effort
    )
    config = "default" if effort == "none" else effort
    try:
        kernel(*args)  # Compile and pick a config outside the capture.
    except helion.exc.BackendUnsupported:
        if effort != "none":
            raise
        # The default matmul lowering (MPP) does not take a single row.
        cfg = dict(kernel.bind(tuple(args)).config_spec.default_config())
        cfg["metal_matmul"] = "fragment"
        config = str(cfg)
        kernel = helion.kernel(
            fn, backend="metal", static_shapes=True, config=helion.Config(**cfg)
        )
        kernel(*args)
    launches = _capture(kernel, args)
    if len(launches) != 1:
        raise NotImplementedError(f"{name} makes {len(launches)} launches")
    spec, ins, outs = _to_spec(name, launches[0])
    check(
        spec,
        {n: np.array(t.cpu()) for n, t in zip(spec["input_names"], ins)},
        [np.array(t.cpu()) for t in outs],
    )
    spec["helion"] = {"version": importlib.metadata.version("helion"), "config": config}
    return spec
