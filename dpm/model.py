import torch
import torch.nn as nn

from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead

from .aggregator import Aggregator
from .decoder import Decoder


def freeze_all_params(modules):
    for module in modules:
        try:
            for n, param in module.named_parameters():
                param.requires_grad = False
        except AttributeError:
            # module is directly a parameter
            module.requires_grad = False


class VDPM(nn.Module):
    def __init__(self, cfg, img_size=518, patch_size=14, embed_dim=1024):
        super().__init__()
        self.cfg = cfg

        self.aggregator = Aggregator(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )
        self.decoder = Decoder(
            cfg,
            dim_in=2*embed_dim,
            embed_dim=embed_dim,
            depth=cfg.model.decoder_depth
        )
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")

        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.set_freeze()

    def set_freeze(self):
        to_be_frozen = [self.aggregator.patch_embed]
        freeze_all_params(to_be_frozen)

    def forward(
        self,
        views, autocast_dpt=None
    ):
        images = torch.stack([view["img"] for view in views], dim=1)
        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        res_dynamic = dict()

        if self.decoder is not None:
            cond_view_idxs = torch.stack([view["view_idxs"][:, 1] for view in views], dim=1)
            decoded_tokens = self.decoder(images, aggregated_tokens_list, patch_start_idx, cond_view_idxs)

        if autocast_dpt is None:
            autocast_dpt = torch.amp.autocast("cuda", enabled=False)

        with autocast_dpt:
            pts3d, pts3d_conf = self.point_head(
                aggregated_tokens_list, images, patch_start_idx
            )

            padded_decoded_tokens = [None] * len(aggregated_tokens_list)
            for idx, layer_idx in enumerate(self.point_head.intermediate_layer_idx):
                padded_decoded_tokens[layer_idx] = decoded_tokens[idx]
            pts3d_dyn, pts3d_dyn_conf = self.point_head(
                padded_decoded_tokens, images, patch_start_idx
            )

            res_dynamic |= {
                "pts3d": pts3d_dyn,
                "conf": pts3d_dyn_conf
            }

            pose_enc_list = self.camera_head(aggregated_tokens_list)
            res_dynamic |= {"pose_enc_list": pose_enc_list}

        res_static = dict(
            pts3d=pts3d,
            conf=pts3d_conf
        )
        return res_static, res_dynamic

    def inference(
        self,
        views,
        images=None
    ):
        autocast_amp = torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16)

        if images is None:
            images = torch.stack([view["img"] for view in views], dim=1)

        with autocast_amp:
            aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        S = images.shape[1]

        predictions = dict()
        pointmaps = []
        ones = torch.ones(1, S, dtype=torch.int64)
        for time_ in range(S):
            cond_view_idxs = ones * time_

            with autocast_amp:
                decoded_tokens = self.decoder(images, aggregated_tokens_list, patch_start_idx, cond_view_idxs)
            padded_decoded_tokens = [None] * len(aggregated_tokens_list)
            for idx, layer_idx in enumerate(self.point_head.intermediate_layer_idx):
                padded_decoded_tokens[layer_idx] = decoded_tokens[idx]

            pts3d, pts3d_conf = self.point_head(
                padded_decoded_tokens, images, patch_start_idx
            )

            pointmaps.append(dict(
                pts3d=pts3d,
                conf=pts3d_conf
            ))

        pose_enc_list = self.camera_head(aggregated_tokens_list)
        predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
        predictions["pose_enc_list"] = pose_enc_list
        predictions["pointmaps"] = pointmaps
        return predictions

    def load_state_dict(self, ckpt, is_VGGT_static=False, **kw):
        # don't load these VGGT heads as not needed
        exclude = ["depth_head", "track_head"]
        ckpt = {k:v for k, v in ckpt.items() if k.split('.')[0] not in exclude}
        return super().load_state_dict(ckpt, **kw)
