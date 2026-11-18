# Copyright © 2026 Apple Inc.

import importlib.util
import unittest

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.kernels import helion

ROWS = [1, 4, 8, 16, 32]
HAS_HELION = all(importlib.util.find_spec(m) for m in ("torch", "helion"))


def rel_err(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return (mx.abs(a - b).max() / mx.abs(b).max()).item()


class TestUseHelion(unittest.TestCase):
    def test_use_helion(self):
        on = helion.enabled()
        helion.set_enabled(False)
        self.assertFalse(helion.use_helion())
        helion.set_enabled(True)
        # Without torch and Helion, the model uses its original code.
        self.assertEqual(helion.use_helion(), HAS_HELION)
        helion.set_enabled(on)


@unittest.skipUnless(
    mx.metal.is_available() and HAS_HELION,
    "Helion kernels need Metal, torch and Helion",
)
class TestHelionKernels(unittest.TestCase):
    def setUp(self):
        helion.set_enabled(True)
        helion.launches.clear()

    def tearDown(self):
        helion.set_enabled(False)

    def test_quantized_linear(self):
        mx.random.seed(0)
        for group in [32, 64]:
            layer = nn.QuantizedLinear(6656, 4096, bias=False, group_size=group, bits=4)
            layer.set_dtype(mx.float16)
            ref = nn.QuantizedLinear(6656, 4096, bias=False, group_size=group, bits=4)
            ref.update(layer.parameters())
            ref.set_dtype(mx.float32)
            for rows in ROWS:
                with self.subTest(group=group, rows=rows):
                    x = mx.random.normal((1, rows, 6656)).astype(mx.float16)
                    helion.launches.clear()
                    out = helion.quantized_linear(layer, x)
                    mx.eval(out)
                    self.assertEqual(helion.launches, {f"affine_qmm/t{rows}": 1})
                    self.assertEqual(out.shape, (1, rows, 4096))
                    self.assertEqual(out.dtype, mx.float16)
                    # As close to the float32 result as the MLX kernel.
                    want = ref(x.astype(mx.float32))
                    err_mlx = rel_err(layer(x), want)
                    self.assertLessEqual(rel_err(out, want), 1.25 * err_mlx + 1e-4)

    def test_quantized_linear_fallback(self):
        def make(n, dtype):
            layer = nn.QuantizedLinear(6656, n, bias=False, group_size=64, bits=4)
            layer.set_dtype(dtype)
            return layer

        f16 = make(4096, mx.float16)
        x = mx.random.normal((1, 8, 6656))
        # Float32 and bfloat16, more rows than MLX_LM_HELION_MAX_ROWS, and
        # batch size > 1.
        cases = [
            (make(4096, mx.float32), x),
            (make(4096, mx.bfloat16), x.astype(mx.bfloat16)),
            (f16, mx.random.normal((1, 33, 6656)).astype(mx.float16)),
            (f16, mx.random.normal((2, 8, 6656)).astype(mx.float16)),
        ]
        for layer, inp in cases:
            out = helion.quantized_linear(layer, inp)
            self.assertTrue(mx.array_equal(out, layer(inp)).item())
        self.assertEqual(helion.launches, {})

    def test_norm_add_norm(self):
        mx.random.seed(1)
        eps1, eps2 = 1e-8, 1e-5
        for rows in ROWS:
            with self.subTest(rows=rows):
                x = mx.random.normal((1, rows, 6656)) * 4
                res = mx.random.normal((1, rows, 6656)) * 4
                w1 = 1 + 0.3 * mx.random.normal((6656,))
                w2 = 1 + 0.3 * mx.random.normal((6656,))
                args = [a.astype(mx.float16) for a in (x, res, w1, w2)]
                helion.launches.clear()
                h, n = helion.norm_add_norm(
                    args[0], args[1], args[2], eps1, args[3], eps2
                )
                mx.eval(h, n)
                self.assertEqual(helion.launches, {f"norm_residual_norm/t{rows}": 1})
                helion.set_enabled(False)
                h_mlx, n_mlx = helion.norm_add_norm(
                    args[0], args[1], args[2], eps1, args[3], eps2
                )
                h32, n32 = helion.norm_add_norm(x, res, w1, eps1, w2, eps2)
                helion.set_enabled(True)
                for out, mlx, want in [(h, h_mlx, h32), (n, n_mlx, n32)]:
                    self.assertEqual(out.dtype, mx.float16)
                    self.assertLessEqual(
                        rel_err(out, want), 1.1 * rel_err(mlx, want) + 1e-4
                    )

    def test_disabled(self):
        helion.set_enabled(False)
        layer = nn.QuantizedLinear(6656, 4096, bias=False, group_size=64, bits=4)
        layer.set_dtype(mx.float16)
        helion.quantized_linear(layer, mx.zeros((1, 8, 6656), dtype=mx.float16))
        self.assertEqual(helion.launches, {})


if __name__ == "__main__":
    unittest.main()
