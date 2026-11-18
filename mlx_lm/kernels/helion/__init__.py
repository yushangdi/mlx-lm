# Copyright © 2026 Apple Inc.

"""Helion kernels for MLX.

This folder has the Helion source of each kernel. 
On the first call for a new input shape, Helion compiles
the kernel to Metal, and the result is saved in MLX_LM_HELION_CACHE
(default ~/.cache/mlx_lm/helion). Later runs read the cache. To compile,
torch with MPS and Helion with the Metal backend must be installed.

The kernels are off by default. Set MLX_LM_HELION=1 or call
set_enabled(True); use_helion() tells if they are on. Each function uses
MLX ops when the kernel does not apply, for example for other dtypes,
batch size > 1 or more than MLX_LM_HELION_MAX_ROWS rows.
"""

import hashlib
import importlib.util
import json
import os
import warnings
from functools import lru_cache
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

_HERE = Path(__file__).parent
_DTYPES = {"float16": mx.float16, "float32": mx.float32, "int16": mx.int16}

_enabled = os.environ.get("MLX_LM_HELION", "0") == "1"
MAX_ROWS = int(os.environ.get("MLX_LM_HELION_MAX_ROWS", "32"))
# Helion autotune effort: "none" uses the default config of the kernel.
EFFORT = os.environ.get("MLX_LM_HELION_AUTOTUNE", "none")
CACHE = Path(os.environ.get("MLX_LM_HELION_CACHE", "~/.cache/mlx_lm/helion"))

launches: dict = {}
_kernels: dict = {}


def set_enabled(enabled: bool):
    global _enabled
    _enabled = enabled


def enabled() -> bool:
    return _enabled


@lru_cache(maxsize=None)
def _installed() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "helion"))


def use_helion() -> bool:
    """True if the kernels are on (MLX_LM_HELION=1 or set_enabled(True))
    and torch and Helion are installed."""
    return _enabled and _installed()


class _Kernel:
    def __init__(self, spec, tag, label=None):
        self.spec = spec
        self.label = label
        self._kernel = mx.fast.metal_kernel(
            name=f"helion_{spec['name']}_{tag}",
            input_names=spec["input_names"],
            output_names=spec["output_names"],
            source=spec["source"],
            header=spec["header"],
        )

    def __call__(self, **arrays):
        s = self.spec
        if self.label is not None:
            launches[self.label] = launches.get(self.label, 0) + 1
        outs = self._kernel(
            inputs=[arrays[n] for n in s["input_names"]],
            grid=tuple(s["grid"]),
            threadgroup=tuple(s["threadgroup"]),
            output_shapes=[tuple(x) for x in s["output_shapes"]],
            output_dtypes=[_DTYPES[d] for d in s["output_dtypes"]],
        )
        return dict(zip(s["output_names"], outs))


def _check(spec, inputs, expected):
    """The MLX kernel must give the outputs of the Helion launch, bit for bit."""
    k = _Kernel(spec, "check")
    outs = k(**{n: mx.array(a) for n, a in inputs.items()})
    for name, want in zip(spec["output_names"], expected):
        if not np.array_equal(np.array(outs[name]), want):
            raise RuntimeError(f"MLX output {name} differs from Helion's")


def _get(name, rows, inputs, scalars):
    """The kernel `name` for inputs, a list of (name, shape, dtype), and
    scalars, or None if Helion cannot provide it."""
    key = json.dumps([name, inputs, scalars, EFFORT])
    if key in _kernels:
        return _kernels[key]
    source = (_HERE / f"{name}.py").read_text()
    digest = hashlib.sha256((source + key).encode()).hexdigest()[:16]
    path = CACHE.expanduser() / f"{name}-{digest}.json"
    kernel = None
    try:
        if path.exists():
            spec = json.loads(path.read_text())
        else:
            from . import _build

            spec = _build.build(name, inputs, scalars, EFFORT, _check)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(spec))
        kernel = _Kernel(spec, digest, f"{name}/t{rows}")
    except Exception as e:
        warnings.warn(f"Helion kernel {name} {inputs} is not used: {e!r}")
    _kernels[key] = kernel
    return kernel


def _fits(x):
    return (
        use_helion()
        and x.ndim == 3
        and x.shape[0] == 1
        and x.shape[1] <= MAX_ROWS
        and x.dtype == mx.float16
    )


def quantized_linear(layer: nn.Module, x: mx.array) -> mx.array:
    """layer(x) for a QuantizedLinear layer, with the Helion matmul."""
    if (
        not _fits(x)
        or not isinstance(layer, nn.QuantizedLinear)
        or layer.mode != "affine"
        or "bias" in layer
        or layer.scales.dtype != mx.float16
    ):
        return layer(x)
    _, rows, K = x.shape
    # The kernel reads MLX's uint32 words as int16 words.
    w = layer.weight.view(mx.int16)
    kernel = _get(
        "affine_qmm",
        rows,
        [
            ("x", [rows, K], "float16"),
            ("w", list(w.shape), "int16"),
            ("scales", list(layer.scales.shape), "float16"),
            ("biases", list(layer.biases.shape), "float16"),
        ],
        {},
    )
    if kernel is None:
        return layer(x)
    out = kernel(x=x, w=w, scales=layer.scales, biases=layer.biases)["out"]
    return out.reshape(1, rows, -1)


def norm_add_norm(x, res, w1, eps1, w2, eps2):
    """h = res + rms_norm(x, w1, eps1), and rms_norm(h, w2, eps2)."""
    if _fits(x) and all(a.dtype == mx.float16 for a in (res, w1, w2)):
        _, rows, D = x.shape
        kernel = _get(
            "norm_residual_norm",
            rows,
            [
                ("x", [rows, D], "float16"),
                ("res", [rows, D], "float16"),
                ("w1", [D], "float16"),
                ("w2", [D], "float16"),
            ],
            {"eps1": float(eps1), "eps2": float(eps2)},
        )
        if kernel is not None:
            out = kernel(x=x, res=res, w1=w1, w2=w2)
            return out["h_out"].reshape(x.shape), out["n_out"].reshape(x.shape)
    h = res + mx.fast.rms_norm(x, w1, eps1)
    return h, mx.fast.rms_norm(h, w2, eps2)


__all__ = [
    "enabled",
    "launches",
    "norm_add_norm",
    "quantized_linear",
    "set_enabled",
    "use_helion",
]
