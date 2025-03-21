import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import deepspeed
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from open_clip.factory import get_model_config, load_state_dict
from open_clip.model import (
    CLIPTextCfg,
    CLIPVisionCfg,
    _build_text_tower,
    _build_vision_tower,
    convert_to_custom_text_state_dict,
    resize_pos_embed,
    resize_text_pos_embed,
)
from open_clip.transformer import text_global_pool
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled


class OpenCLIPVisionTower(nn.Module):

    def __init__(
        self,
        vision_tower,
        vision_tower_path,
        optimize_vision_tower_aux,
        use_multi_level=False,
        last_only=False,
        delay_load=False,
        use_text=False,
        use_last_feat=False,
    ):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.vision_tower_path = vision_tower_path
        self.vision_config = json.load(
            open(os.path.join(self.vision_tower_path, "open_clip_config.json"), "r")
        )
        self.is_optimize = optimize_vision_tower_aux
        self.use_multi_level = use_multi_level
        self.last_only = last_only
        self.use_text = use_text
        self.use_last_feat = use_last_feat
        if not delay_load:
            self.load_model()

    def load_model(self):
        ckpt_path = os.path.join(self.vision_tower_path, "open_clip_pytorch_model.bin")
        if "convnext" in self.vision_tower_name:
            if "large" in self.vision_tower_name and "d-320" in self.vision_tower_name:
                self.model_type = "convnext_large_d_320"
                self.model_channel = [192, 384, 768, 1536]  # stage 0-3
            elif "base" in self.vision_tower_name and "w-320" in self.vision_tower_name:
                self.model_type = "convnext_base_w_320"
                self.model_channel = [128, 256, 512, 1024]
            elif "xxlarge" in self.vision_tower_name:
                self.model_type = "convnext_xxlarge"
                self.model_channel = [384, 768, 1536, 3072]
            else:
                print(self.vision_tower_name)
                raise

        clip_model = CLIP(**get_model_config(self.model_type), use_text=self.use_text)
        if not self.use_text:
            clip_model.visual.trunk.norm_pre = None
            clip_model.visual.trunk.head = None
            clip_model.visual.head = None
        print(f"Loading pretrained weights ({self.model_type}).")
        load_checkpoint(clip_model, ckpt_path, strict=False)

        self.is_loaded = True
        # decompose stem and stages blocks in vision tower
        self.vision_stem = clip_model.visual.trunk.stem
        self.vision_stages = clip_model.visual.trunk.stages
        if not self.is_optimize:
            self.vision_stem.requires_grad_(False)
            self.vision_stages.requires_grad_(False)

    def forward_after_stage(self, x):
        # trunk
        x = self.vision_trunk_norm_pre(x)
        x = self.vision_trunk_head.global_pool(x)
        x = self.vision_trunk_head.norm(x)
        x = self.vision_trunk_head.flatten(x)
        x = self.vision_trunk_head.drop(x)
        x = self.vision_trunk_head.fc(x)
        # head
        x = self.vision_head(x)

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()
        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        x, _ = text_global_pool(x, text, self.text_pool_type)
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                x = self.text_projection(x)
            else:
                x = x @ self.text_projection

        return F.normalize(x, dim=-1) if normalize else x

    def forward(self, images):

        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.backbone(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                )
                image_features.append(image_feature)
        else:
            image_features = self.backbone(
                images.to(device=self.device, dtype=self.dtype),
            )

        return {
            "image_features": image_features,
            "last_feat": image_features[-1],
        }

    def backbone(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], List[int]]:
        """Process the input images through the backbone network.

        Inputs:
            images (torch.Tensor): The input images.

        Returns:
            Tuple[List[torch.Tensor], List[int]]: A tuple containing a list of feature maps and a
                ist of channels per level.
        """
        if not self.is_optimize:
            with torch.no_grad():
                results = self.basic_forward(images)
        else:
            results = self.basic_forward(images)

        feature_maps = []

        for _stage in results:
            feature_maps.append(results[_stage].contiguous())
        return feature_maps

    def basic_forward(self, images):
        results = {}
        x = self.vision_stem(images)
        for _idx in range(len(self.vision_stages)):
            x = self.vision_stages[_idx](x)
            results[f"stage_{_idx}"] = x
        return results

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_stem[0].weight.dtype

    @property
    def device(self):
        return self.vision_stem[0].weight.device

    @property
    def config(self):
        return self.vision_config

    @property
    def hidden_size(self):
        return sum(self.model_channel)


# modified function from open_clip to support zero3 stage
def load_checkpoint(model, checkpoint_path, strict=True):
    if Path(checkpoint_path).suffix in (".npz", ".npy"):
        from open_clip.big_vision import load_big_vision_weights

        load_big_vision_weights(model, checkpoint_path)
        return {}

    state_dict = load_state_dict(checkpoint_path)
    # detect old format and make compatible with new format
    if "positional_embedding" in state_dict and not hasattr(
        model, "positional_embedding"
    ):
        state_dict = convert_to_custom_text_state_dict(state_dict)
    # If loading a non-SigLIP model for SigLIP training. See https://github.com/mlfoundations/open_clip/issues/712
    # if 'logit_bias' not in state_dict and model.logit_bias is not None:
    #     state_dict["logit_bias"] = torch.zeros_like(state_dict["logit_scale"])
    # Certain text transformers no longer expect position_ids after transformers==4.31
    position_id_key = "text.transformer.embeddings.position_ids"
    if position_id_key in state_dict and not hasattr(model, position_id_key):
        del state_dict[position_id_key]
    resize_pos_embed(state_dict, model)
    resize_text_pos_embed(state_dict, model)
    # resize_text_pos_embed(state_dict, model)
    # incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    if is_deepspeed_zero3_enabled():

        error_msgs = []

        def load(module: nn.Module, state_dict, prefix=""):
            metadata = None

            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)
            # Parameters of module and children will start with prefix. We can exit early if there are none in this
            # state_dict
            if len([key for key in state_dict if key.startswith(prefix)]) > 0:
                if is_deepspeed_zero3_enabled():
                    # In sharded models, each shard has only part of the full state_dict, so only gather
                    # parameters that are in the current state_dict.
                    named_parameters = dict(
                        module.named_parameters(prefix=prefix[:-1], recurse=False)
                    )
                    params_to_gather = [
                        named_parameters[k]
                        for k in state_dict.keys()
                        if k in named_parameters
                    ]
                    if len(params_to_gather) > 0:
                        # because zero3 puts placeholders in model params, this context
                        # manager gathers (unpartitions) the params of the current layer, then loads from
                        # the state dict and then re-partitions them again
                        with deepspeed.zero.GatheredParameters(
                            params_to_gather, modifier_rank=0
                        ):
                            if torch.distributed.get_rank() == 0:
                                module._load_from_state_dict(*args)
                else:
                    module._load_from_state_dict(*args)

            for name, child in module._modules.items():
                if child is not None:
                    load(child, state_dict, prefix + name + ".")

        load(model, state_dict)
        incompatible_keys = []
    else:
        incompatible_keys = model.load_state_dict(state_dict, strict=strict)
        logging.info(
            f"incompatible_keys.missing_keys: {incompatible_keys.missing_keys}"
        )
    return incompatible_keys


class CLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
        self,
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        use_text=False,
        cast_dtype: Optional[torch.dtype] = None,
        output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict

        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        if use_text:
            text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
            self.transformer = text.transformer
            self.context_length = text.context_length
            self.vocab_size = text.vocab_size
            self.token_embedding = text.token_embedding
            self.positional_embedding = text.positional_embedding
            self.ln_final = text.ln_final
            self.text_projection = text.text_projection
            self.text_pool_type = text.pool_type
            self.register_buffer("attn_mask", text.attn_mask, persistent=False)
            import numpy as np

            init_logit_scale = (np.log(1 / 0.07),)
            init_logit_bias = (None,)
            self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
