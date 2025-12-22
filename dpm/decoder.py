# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE-VGGT file in the root directory of this source tree.

import logging
import torch
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint
from typing import List, Callable
from dataclasses import dataclass

from einops import repeat

from vggt.layers.block import drop_add_residual_stochastic_depth
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter

from vggt.layers.attention import Attention
from vggt.layers.drop_path import DropPath
from vggt.layers.layer_scale import LayerScale
from vggt.layers.mlp import Mlp

logger = logging.getLogger(__name__)


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class ConditionalBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim, elementwise_affine=False)
        self.modulation = Modulation(dim, double=False)

        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            fused_attn=fused_attn,
            rope=rope,
        )

        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor, pos=None, cond=None, is_global=False) -> Tensor:
        B, S = cond.shape[:2]
        C = x.shape[-1]
        if is_global:
            P = x.shape[1] // S
        cond = cond.view(B * S, C)
        mod, _ = self.modulation(cond)

        def attn_residual_func(x: Tensor, pos=None) -> Tensor:
            """
            conditional attention following DiT implementation from Flux
            https://github.com/black-forest-labs/flux/blob/main/src/flux/modules/layers.py#L194-L239
            """
            def prepare_for_mod(y):
                """reshape to modulate the patch tokens with correct conditioning one"""
                return y.view(B, S, P, C).view(B * S, P, C) if is_global else y
            def restore_after_mod(y):
                """reshape back to global sequence"""
                return y.view(B, S, P, C).view(B, S * P, C) if is_global else y

            x = prepare_for_mod(x)
            x = (1 + mod.scale) * self.norm1(x) + mod.shift
            x = restore_after_mod(x)

            x = self.attn(x, pos=pos)

            x = prepare_for_mod(x)
            x = mod.gate * x
            x = restore_after_mod(x)

            return x

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x,
                pos=pos,
                residual_func=attn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
            x = drop_add_residual_stochastic_depth(
                x,
                residual_func=ffn_residual_func,
                sample_drop_ratio=self.sample_drop_ratio,
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x, pos=pos))
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x = x + attn_residual_func(x, pos=pos)
            x = x + ffn_residual_func(x)
        return x


class Decoder(nn.Module):
    """Attention blocks after encoder per DPT input feature
    to generate point maps at a given time.
    """

    def __init__(
        self,
        cfg,
        dim_in: int,
        intermediate_layer_idx: List[int] = [4, 11, 17, 23],
        patch_size=14,
        embed_dim=1024,
        depth=2,
        num_heads=16,
        mlp_ratio=4.0,
        block_fn=ConditionalBlock,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
    ):
        super().__init__()
        self.cfg = cfg
        self.intermediate_layer_idx = intermediate_layer_idx

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        self.rope = (
            RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        )
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.dim_in = dim_in

        self.old_decoder = False
        if self.old_decoder:
            self.frame_blocks = nn.ModuleList(
                [
                    block_fn(
                        dim=embed_dim*2,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        proj_bias=proj_bias,
                        ffn_bias=ffn_bias,
                        init_values=init_values,
                        qk_norm=qk_norm,
                        rope=self.rope,
                    )
                    for _ in range(depth)
                ]
            )
            self.global_blocks = nn.ModuleList(
                [
                    block_fn(
                        dim=embed_dim*2,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        proj_bias=proj_bias,
                        ffn_bias=ffn_bias,
                        init_values=init_values,
                        qk_norm=qk_norm,
                        rope=self.rope,
                    )
                    for _ in range(depth)
                ]
            )
        else:
            depths = [depth]
            self.frame_blocks = nn.ModuleList([
                nn.ModuleList([
                    block_fn(
                        dim=embed_dim*2,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        proj_bias=proj_bias,
                        ffn_bias=ffn_bias,
                        init_values=init_values,
                        qk_norm=qk_norm,
                        rope=self.rope,
                    )
                    for _ in range(d)
                ])
                for d in depths
            ])

            self.global_blocks = nn.ModuleList([
                nn.ModuleList([
                    block_fn(
                        dim=embed_dim*2,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        proj_bias=proj_bias,
                        ffn_bias=ffn_bias,
                        init_values=init_values,
                        qk_norm=qk_norm,
                        rope=self.rope,
                    )
                    for _ in range(d)
                ])
                for d in depths
            ])

        self.use_reentrant = False # hardcoded to False

    def get_condition_tokens(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        cond_view_idxs: torch.Tensor
    ):
        # Use tokens from the last block for conditioning
        tokens_last = aggregated_tokens_list[-1]  # [B S N_tok D]
        # Extract the camera tokens
        cond_token_idx = 1
        camera_tokens = tokens_last[:, :, [cond_token_idx]]  # [B S D]

        cond_view_idxs = cond_view_idxs.to(camera_tokens.device)
        cond_view_idxs = repeat(
            cond_view_idxs,
            "b s -> b s c d",
            c=camera_tokens.shape[2],
            d=camera_tokens.shape[3],
        )
        cond_tokens = torch.gather(camera_tokens, 1, cond_view_idxs)

        return cond_tokens

    def forward(
        self,
        images: torch.Tensor,
        aggregated_tokens_list: List[torch.Tensor],
        patch_start_idx: int,
        cond_view_idxs: torch.Tensor,
    ):
        B, S, _, H, W = images.shape

        cond_tokens = self.get_condition_tokens(
            aggregated_tokens_list, cond_view_idxs
        )

        input_tokens = []
        for k, layer_idx in enumerate(self.intermediate_layer_idx):
            layer_tokens = aggregated_tokens_list[layer_idx].clone()
            input_tokens.append(layer_tokens)

        _, _, P, C = input_tokens[0].shape

        pos = None
        if self.rope is not None:
            pos = self.position_getter(
                B * S, H // self.patch_size, W // self.patch_size, device=images.device
            )
        if patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        frame_idx = 0
        global_idx = 0
        depth = len(self.frame_blocks[0])
        N = len(input_tokens)
        # stack all intermediate layer tokens along batch dimension
        # they are all processed by the same decoder
        s_tokens = torch.cat(input_tokens)
        s_cond_tokens = torch.cat([cond_tokens] * N, dim=0)
        s_pos = torch.cat([pos] * N, dim=0)

        # perform time conditioned attention
        for _ in range(depth):
            for attn_type in self.aa_order:
                token_idx = 0

                if attn_type == "frame":
                    s_tokens, frame_idx, _ = self._process_frame_attention(
                        s_tokens, s_cond_tokens, B * N, S, P, C, frame_idx, pos=s_pos, token_idx=token_idx
                    )
                elif attn_type == "global":
                    s_tokens, global_idx, _ = self._process_global_attention(
                        s_tokens, s_cond_tokens, B * N, S, P, C, global_idx, pos=s_pos, token_idx=token_idx
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")
        processed = [t.view(B, S, P, C) for t in s_tokens.split(B, dim=0)]

        return processed

    def _process_frame_attention(self, tokens, cond_tokens, B, S, P, C, frame_idx, pos=None, token_idx=0):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []
        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.frame_blocks[token_idx][frame_idx], tokens, pos, cond_tokens, use_reentrant=self.use_reentrant)
            else:
                if self.old_decoder:
                    tokens = self.frame_blocks[frame_idx](tokens, pos=pos, cond=cond_tokens)
                else:
                    tokens = self.frame_blocks[0][frame_idx](tokens, pos=pos, cond=cond_tokens)

            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, cond_tokens, B, S, P, C, global_idx, pos=None, token_idx=0):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.global_blocks[token_idx][global_idx], tokens, pos, cond_tokens, True, use_reentrant=self.use_reentrant)
            else:
                if self.old_decoder:
                    tokens = self.global_blocks[global_idx](tokens, pos=pos, cond=cond_tokens, is_global=True)
                else:
                    tokens = self.global_blocks[0][global_idx](tokens, pos=pos, cond=cond_tokens, is_global=True)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates
