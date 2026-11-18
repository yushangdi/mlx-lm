# Copyright © 2026 Apple Inc.

"""Helion kernel: RMSNorm, residual add, RMSNorm.
"""

import helion.language as hl
import torch

F32 = torch.float32


def norm_residual_norm(
    x: torch.Tensor,
    res: torch.Tensor,
    w1: torch.Tensor,
    eps1: float,
    w2: torch.Tensor,
    eps2: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """h = res + rms_norm(x) * w1 and rms_norm(h) * w2, for x and res [M, D].
    The values round to the dtype of x where the MLX ops round."""
    m, d = x.size()
    dt = x.dtype
    h_out = torch.empty_like(res)
    n_out = torch.empty_like(res)
    for tile_m in hl.tile(m):
        xf = x[tile_m, :].to(F32)
        inv = torch.rsqrt(torch.sum(xf * xf, -1) / d + eps1)
        normed = (w1[None, :].to(F32) * (xf * inv[:, None]).to(dt).to(F32)).to(dt)
        hb = (res[tile_m, :].to(F32) + normed.to(F32)).to(dt)
        h_out[tile_m, :] = hb
        hf = hb.to(F32)
        inv2 = torch.rsqrt(torch.sum(hf * hf, -1) / d + eps2)
        n_out[tile_m, :] = (
            w2[None, :].to(F32) * (hf * inv2[:, None]).to(dt).to(F32)
        ).to(dt)
    return h_out, n_out
