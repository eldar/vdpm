from typing import Tuple, Dict, Set

import torch
from torch import Tensor
from jaxtyping import Float, Bool, UInt8, Int32


def inside_image(
    pts2d: Float[Tensor, "n 2"],
    image_size: Tuple[int, ...]
) -> Float[Tensor, " n"]:
    H, W = image_size
    px, py = pts2d.unbind(-1)
    return (
        (0 <= px) & (px < W) &
        (0 <= py) & (py < H)
    )


def get_uv_grid(
    image_size: Tuple[int, int],
    dtype=torch.float32
) -> Float[Tensor, "h w 2"]:
    H, W = image_size
    meshgrid = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
    id_coords = torch.stack(meshgrid, dim=-1).to(dtype)
    return id_coords


def persp_project(xyz):
    z = xyz[:, 2:]
    uv = xyz[:, :2] / z
    return uv, z


