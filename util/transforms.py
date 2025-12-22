import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from jaxtyping import Float


def transform_points(
    T: Float[Tensor, "... d d"],
    pts: Float[Tensor, "... n c"]
) -> Float[Tensor, "... n 3"]:
    """
    Args:
        T (torch.Tensor): transformation matrix of shape (d, d)
        pts (torch.Tensor): Input points of shape (n, c)
    """
    if pts.shape[-1] == (T.shape[-1] - 1):
        pts = F.pad(pts, (0, 1), value=1)
    pts = torch.einsum("...ji,...ni->...nj", T, pts)
    return pts[..., :3]


def transform_points_np(
    T: Float[np.ndarray, "... d d"],
    pts: Float[np.ndarray, "... n c"]
) -> Float[np.ndarray, "... n 3"]:
    """
    Args:
        T (torch.Tensor): transformation matrix of shape (d, d)
        pts (torch.Tensor): Input points of shape (n, c)
    """
    orig_shape = pts.shape
    pts = pts.reshape(-1, 3)
    if pts.shape[-1] == (T.shape[-1] - 1):
        pts = np.pad(pts, ((0, 0), (0, 1)), constant_values=1)
    pts = np.einsum("...ji,...ni->...nj", T, pts)
    pts = pts[..., :3]
    pts = pts.reshape(orig_shape)
    return pts


def invert_intrinsics(
    K: Float[Tensor, "3 3"]
):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    K_inv = torch.tensor([
        [1/fx, 0,    -cx/fx],
        [0,    1/fy, -cy/fy],
        [0,    0,    1     ]
    ], device=K.device)
    return K_inv


def se3_from_Rt(
    R: Float[Tensor, "3 3"],
    t: Float[Tensor, "3"]
) -> Float[Tensor, "4 4"]:
    T = torch.eye(4, dtype=R.dtype)
    T[:3, :3] = R
    T[:3,  3] = t
    return T


def invert_se3(
    T: Float[Tensor, "4 4"]
):
    R_ = T[:3, :3].transpose(0, 1)
    t = T[:3, 3]
    t_ = -torch.einsum("ij,j->i", R_, t)
    T_ = torch.eye(4, dtype=T.dtype)
    T_[:3, :3] = R_
    T_[:3, 3] = t_
    return T_


def to_4x4(
    m: Float[Tensor, "3 3"]
):
    m_ = torch.eye(4, dtype=m.dtype)
    m_[:3, :3] = m
    return m_


def project_points(
    K: Float[Tensor, "... d d"],
    pts: Float[Tensor, "... n c"]
):
    """
    Non-differentiable
    """
    if K.shape[-1] == 3:
        K = to_4x4(K)
    xyz = transform_points(K, pts)
    uv = xyz[..., :2] / xyz[..., 2:]
    return uv

