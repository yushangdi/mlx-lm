# Copyright © 2026 Apple Inc.

"""Helion kernel: x @ W.T for W in MLX's affine quantized format.
"""

import helion.language as hl
import torch

F32 = torch.float32


def affine_qmm(
    x: torch.Tensor, w: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor
) -> torch.Tensor:
    """W[n, k] = scales[n, g] * q + biases[n, g], g = k // group. The codes are
    MLX's uint32 words, given as int16 words w [N, K * bits / 16]. scales and
    biases are [N, K / group]. The bit width and the group size come from the
    shapes. The product uses fp32 operands and an fp32 accumulator."""
    m, k = x.size()
    n, kw = w.size()
    bits = 16 * kw // k
    group = k // scales.size(1)
    mask = (1 << bits) - 1
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=F32)
        for tile_k in hl.tile(k):
            bit = tile_k.index * bits
            lo = w[tile_n, bit // 16].to(torch.int32) & 0xFFFF
            if 16 % bits == 0:
                q = (lo >> (bit % 16)[None, :]) & mask
            else:
                # A code that crosses two int16 words reads both.
                nxt = torch.clamp(bit // 16 + 1, max=kw - 1)
                hi = w[tile_n, nxt].to(torch.int32) & 0xFFFF
                q = ((lo | (hi << 16)) >> (bit % 16)[None, :]) & mask
            grp = tile_k.index // group
            deq = q.to(F32) * scales[tile_n, grp].to(F32) + biases[tile_n, grp].to(F32)
            # mx.quantized_matmul does not round the weight to the dtype of x.
            acc = torch.addmm(acc, x[tile_m, tile_k].to(F32), deq.T)
        out[tile_m, tile_n] = acc.to(x.dtype)
    return out
