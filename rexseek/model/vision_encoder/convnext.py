from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from open_clip.factory import get_model_config
from open_clip.model import CLIPVisionCfg
from timm.layers import (
    AvgPool2dSame,
    ClassifierHead,
    DropPath,
    GlobalResponseNormMlp,
    LayerNorm,
    LayerNorm2d,
    Mlp,
    NormMlpClassifierHead,
    create_conv2d,
    get_act_layer,
    make_divisible,
    to_ntuple,
    trunc_normal_,
)
from timm.models._builder import build_model_with_cfg
from timm.models._features import feature_take_indices
from timm.models._manipulate import checkpoint_seq, named_apply

__all__ = ["ConvNeXt"]  # model_registry will add each entrypoint fn to this


class Downsample(nn.Module):

    def __init__(self, in_chs, out_chs, stride=1, dilation=1):
        super().__init__()
        avg_stride = stride if dilation == 1 else 1
        if stride > 1 or dilation > 1:
            avg_pool_fn = (
                AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
            )
            self.pool = avg_pool_fn(
                2, avg_stride, ceil_mode=True, count_include_pad=False
            )
        else:
            self.pool = nn.Identity()

        if in_chs != out_chs:
            self.conv = create_conv2d(in_chs, out_chs, 1, stride=1)
        else:
            self.conv = nn.Identity()

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block
    There are two equivalent implementations:
      (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
      (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back

    Unlike the official impl, this one allows choice of 1 or 2, 1x1 conv can be faster with appropriate
    choice of LayerNorm impl, however as model size increases the tradeoffs appear to change and nn.Linear
    is a better choice. This was observed with PyTorch 1.10 on 3090 GPU, it could change over time & w/ different HW.
    """

    def __init__(
        self,
        in_chs: int,
        out_chs: Optional[int] = None,
        kernel_size: int = 7,
        stride: int = 1,
        dilation: Union[int, Tuple[int, int]] = (1, 1),
        mlp_ratio: float = 4,
        conv_mlp: bool = False,
        conv_bias: bool = True,
        use_grn: bool = False,
        ls_init_value: Optional[float] = 1e-6,
        act_layer: Union[str, Callable] = "gelu",
        norm_layer: Optional[Callable] = None,
        drop_path: float = 0.0,
    ):
        """

        Args:
            in_chs: Block input channels.
            out_chs: Block output channels (same as in_chs if None).
            kernel_size: Depthwise convolution kernel size.
            stride: Stride of depthwise convolution.
            dilation: Tuple specifying input and output dilation of block.
            mlp_ratio: MLP expansion ratio.
            conv_mlp: Use 1x1 convolutions for MLP and a NCHW compatible norm layer if True.
            conv_bias: Apply bias for all convolution (linear) layers.
            use_grn: Use GlobalResponseNorm in MLP (from ConvNeXt-V2)
            ls_init_value: Layer-scale init values, layer-scale applied if not None.
            act_layer: Activation layer.
            norm_layer: Normalization layer (defaults to LN if not specified).
            drop_path: Stochastic depth probability.
        """
        super().__init__()
        out_chs = out_chs or in_chs
        dilation = to_ntuple(2)(dilation)
        act_layer = get_act_layer(act_layer)
        if not norm_layer:
            norm_layer = LayerNorm2d if conv_mlp else LayerNorm
        mlp_layer = partial(
            GlobalResponseNormMlp if use_grn else Mlp, use_conv=conv_mlp
        )
        self.use_conv_mlp = conv_mlp
        self.conv_dw = create_conv2d(
            in_chs,
            out_chs,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation[0],
            depthwise=True,
            bias=conv_bias,
        )
        self.norm = norm_layer(out_chs)
        self.mlp = mlp_layer(out_chs, int(mlp_ratio * out_chs), act_layer=act_layer)
        self.ramma = (
            nn.Parameter(ls_init_value * torch.ones(out_chs))
            if ls_init_value is not None
            else None
        )
        if in_chs != out_chs or stride != 1 or dilation[0] != dilation[1]:
            self.shortcut = Downsample(
                in_chs, out_chs, stride=stride, dilation=dilation[0]
            )
        else:
            self.shortcut = nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        if self.use_conv_mlp:
            x = self.norm(x)
            x = self.mlp(x)
        else:
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
            x = self.mlp(x)
            x = x.permute(0, 3, 1, 2)
        if self.ramma is not None:
            x = x.mul(self.ramma.reshape(1, -1, 1, 1))

        x = self.drop_path(x) + self.shortcut(shortcut)
        return x


class ConvNeXtStage(nn.Module):

    def __init__(
        self,
        in_chs,
        out_chs,
        kernel_size=7,
        stride=2,
        depth=2,
        dilation=(1, 1),
        drop_path_rates=None,
        ls_init_value=1.0,
        conv_mlp=False,
        conv_bias=True,
        use_grn=False,
        act_layer="gelu",
        norm_layer=None,
        norm_layer_cl=None,
    ):
        super().__init__()
        self.grad_checkpointing = False

        if in_chs != out_chs or stride > 1 or dilation[0] != dilation[1]:
            ds_ks = 2 if stride > 1 or dilation[0] != dilation[1] else 1
            pad = (
                "same" if dilation[1] > 1 else 0
            )  # same padding needed if dilation used
            self.downsample = nn.Sequential(
                norm_layer(in_chs),
                create_conv2d(
                    in_chs,
                    out_chs,
                    kernel_size=ds_ks,
                    stride=stride,
                    dilation=dilation[0],
                    padding=pad,
                    bias=conv_bias,
                ),
            )
            in_chs = out_chs
        else:
            self.downsample = nn.Identity()

        drop_path_rates = drop_path_rates or [0.0] * depth
        stage_blocks = []
        for i in range(depth):
            stage_blocks.append(
                ConvNeXtBlock(
                    in_chs=in_chs,
                    out_chs=out_chs,
                    kernel_size=kernel_size,
                    dilation=dilation[1],
                    drop_path=drop_path_rates[i],
                    ls_init_value=ls_init_value,
                    conv_mlp=conv_mlp,
                    conv_bias=conv_bias,
                    use_grn=use_grn,
                    act_layer=act_layer,
                    norm_layer=norm_layer if conv_mlp else norm_layer_cl,
                )
            )
            in_chs = out_chs
        self.blocks = nn.Sequential(*stage_blocks)

    def forward(self, x):
        x = self.downsample(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        return x


class ConvNeXt(nn.Module):
    r"""ConvNeXt
    A PyTorch impl of : `A ConvNet for the 2020s`  - https://arxiv.org/pdf/2201.03545.pdf
    """

    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: str = "avg",
        output_stride: int = 32,
        depths: Tuple[int, ...] = (3, 3, 9, 3),
        dims: Tuple[int, ...] = (96, 192, 384, 768),
        kernel_sizes: Union[int, Tuple[int, ...]] = 7,
        ls_init_value: Optional[float] = 1e-6,
        stem_type: str = "patch",
        patch_size: int = 4,
        head_init_scale: float = 1.0,
        head_norm_first: bool = False,
        head_hidden_size: Optional[int] = None,
        conv_mlp: bool = False,
        conv_bias: bool = True,
        use_grn: bool = False,
        act_layer: Union[str, Callable] = "gelu",
        norm_layer: Optional[Union[str, Callable]] = None,
        norm_eps: Optional[float] = None,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        """
        Args:
            in_chans: Number of input image channels.
            num_classes: Number of classes for classification head.
            global_pool: Global pooling type.
            output_stride: Output stride of network, one of (8, 16, 32).
            depths: Number of blocks at each stage.
            dims: Feature dimension at each stage.
            kernel_sizes: Depthwise convolution kernel-sizes for each stage.
            ls_init_value: Init value for Layer Scale, disabled if None.
            stem_type: Type of stem.
            patch_size: Stem patch size for patch stem.
            head_init_scale: Init scaling value for classifier weights and biases.
            head_norm_first: Apply normalization before global pool + head.
            head_hidden_size: Size of MLP hidden layer in head if not None and head_norm_first == False.
            conv_mlp: Use 1x1 conv in MLP, improves speed for small networks w/ chan last.
            conv_bias: Use bias layers w/ all convolutions.
            use_grn: Use Global Response Norm (ConvNeXt-V2) in MLP.
            act_layer: Activation layer type.
            norm_layer: Normalization layer type.
            drop_rate: Head pre-classifier dropout rate.
            drop_path_rate: Stochastic depth drop rate.
        """
        super().__init__()
        assert output_stride in (8, 16, 32)
        kernel_sizes = to_ntuple(4)(kernel_sizes)
        if norm_layer is None:
            norm_layer = LayerNorm2d
            norm_layer_cl = norm_layer if conv_mlp else LayerNorm
            if norm_eps is not None:
                norm_layer = partial(norm_layer, eps=norm_eps)
                norm_layer_cl = partial(norm_layer_cl, eps=norm_eps)
        else:
            assert (
                conv_mlp
            ), "If a norm_layer is specified, conv MLP must be used so all norm expect rank-4, channels-first input"
            norm_layer_cl = norm_layer
            if norm_eps is not None:
                norm_layer_cl = partial(norm_layer_cl, eps=norm_eps)

        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.feature_info = []

        assert stem_type in ("patch", "overlap", "overlap_tiered")
        if stem_type == "patch":
            # NOTE: this stem is a minimal form of ViT PatchEmbed, as used in SwinTransformer w/ patch_size = 4
            self.stem = nn.Sequential(
                nn.Conv2d(
                    in_chans,
                    dims[0],
                    kernel_size=patch_size,
                    stride=patch_size,
                    bias=conv_bias,
                ),
                norm_layer(dims[0]),
            )
            stem_stride = patch_size
        else:
            mid_chs = make_divisible(dims[0] // 2) if "tiered" in stem_type else dims[0]
            self.stem = nn.Sequential(
                nn.Conv2d(
                    in_chans,
                    mid_chs,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=conv_bias,
                ),
                nn.Conv2d(
                    mid_chs, dims[0], kernel_size=3, stride=2, padding=1, bias=conv_bias
                ),
                norm_layer(dims[0]),
            )
            stem_stride = 4

        self.stages = nn.Sequential()
        dp_rates = [
            x.tolist()
            for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)
        ]
        stages = []
        prev_chs = dims[0]
        curr_stride = stem_stride
        dilation = 1
        # 4 feature resolution stages, each consisting of multiple residual blocks
        for i in range(4):
            stride = 2 if curr_stride == 2 or i > 0 else 1
            if curr_stride >= output_stride and stride > 1:
                dilation *= stride
                stride = 1
            curr_stride *= stride
            first_dilation = 1 if dilation in (1, 2) else 2
            out_chs = dims[i]
            stages.append(
                ConvNeXtStage(
                    prev_chs,
                    out_chs,
                    kernel_size=kernel_sizes[i],
                    stride=stride,
                    dilation=(first_dilation, dilation),
                    depth=depths[i],
                    drop_path_rates=dp_rates[i],
                    ls_init_value=ls_init_value,
                    conv_mlp=conv_mlp,
                    conv_bias=conv_bias,
                    use_grn=use_grn,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    norm_layer_cl=norm_layer_cl,
                )
            )
            prev_chs = out_chs
            # NOTE feature_info use currently assumes stage 0 == stride 1, rest are stride 2
            self.feature_info += [
                dict(num_chs=prev_chs, reduction=curr_stride, module=f"stages.{i}")
            ]
        self.stages = nn.Sequential(*stages)
        self.num_features = self.head_hidden_size = prev_chs

        # if head_norm_first == true, norm -> global pool -> fc ordering, like most other nets
        # otherwise pool -> norm -> fc, the default ConvNeXt ordering (pretrained FB weights)
        if head_norm_first:
            assert not head_hidden_size
            self.norm_pre = norm_layer(self.num_features)
            self.head = ClassifierHead(
                self.num_features,
                num_classes,
                pool_type=global_pool,
                drop_rate=self.drop_rate,
            )
        else:
            self.norm_pre = nn.Identity()
            self.head = NormMlpClassifierHead(
                self.num_features,
                num_classes,
                hidden_size=head_hidden_size,
                pool_type=global_pool,
                drop_rate=self.drop_rate,
                norm_layer=norm_layer,
                act_layer="gelu",
            )
            self.head_hidden_size = self.head.num_features
        named_apply(partial(_init_weights, head_init_scale=head_init_scale), self)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r"^stem",
            blocks=(
                r"^stages\.(\d+)"
                if coarse
                else [
                    (r"^stages\.(\d+)\.downsample", (0,)),  # blocks
                    (r"^stages\.(\d+)\.blocks\.(\d+)", None),
                    (r"^norm_pre", (99999,)),
                ]
            ),
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head.fc

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        self.head.reset(num_classes, global_pool)

    def forward_intermediates(
        self,
        x: torch.Tensor,
        indices: Optional[Union[int, List[int], Tuple[int]]] = None,
        norm: bool = False,
        stop_early: bool = False,
        output_fmt: str = "NCHW",
        intermediates_only: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            norm: Apply norm layer to compatible intermediates
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs
            intermediates_only: Only return intermediate features
        Returns:

        """
        assert output_fmt in ("NCHW",), "Output shape must be NCHW."
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.stages) + 1, indices)

        # forward pass
        feat_idx = 0  # stem is index 0
        x = self.stem(x)
        if feat_idx in take_indices:
            intermediates.append(x)

        if (
            torch.jit.is_scripting() or not stop_early
        ):  # can't slice blocks in torchscript
            stages = self.stages
        else:
            stages = self.stages[:max_index]
        for stage in stages:
            feat_idx += 1
            x = stage(x)
            if feat_idx in take_indices:
                # NOTE not bothering to apply norm_pre when norm=True as almost no models have it enabled
                intermediates.append(x)

        if intermediates_only:
            return intermediates

        x = self.norm_pre(x)

        return x, intermediates

    def prune_intermediate_layers(
        self,
        indices: Union[int, List[int], Tuple[int]] = 1,
        prune_norm: bool = False,
        prune_head: bool = True,
    ):
        """Prune layers not required for specified intermediates."""
        take_indices, max_index = feature_take_indices(len(self.stages) + 1, indices)
        self.stages = self.stages[:max_index]  # truncate blocks w/ stem as idx 0
        if prune_norm:
            self.norm_pre = nn.Identity()
        if prune_head:
            self.reset_classifier(0, "")
        return take_indices

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.norm_pre(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=True) if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _init_weights(module, name=None, head_init_scale=1.0):
    if isinstance(module, nn.Conv2d):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        nn.init.zeros_(module.bias)
        if name and "head." in name:
            module.weight.data.mul_(head_init_scale)
            module.bias.data.mul_(head_init_scale)


def checkpoint_filter_fn(state_dict, model):
    """Remap FB checkpoints -> timm"""
    if "head.norm.weight" in state_dict or "norm_pre.weight" in state_dict:
        return state_dict  # non-FB checkpoint
    if "model" in state_dict:
        state_dict = state_dict["model"]

    out_dict = {}
    if "visual.trunk.stem.0.weight" in state_dict:
        out_dict = {
            k.replace("visual.trunk.", ""): v
            for k, v in state_dict.items()
            if k.startswith("visual.trunk.")
        }
        if "visual.head.proj.weight" in state_dict:
            out_dict["head.fc.weight"] = state_dict["visual.head.proj.weight"]
            out_dict["head.fc.bias"] = torch.zeros(
                state_dict["visual.head.proj.weight"].shape[0]
            )
        elif "visual.head.mlp.fc1.weight" in state_dict:
            out_dict["head.pre_logits.fc.weight"] = state_dict[
                "visual.head.mlp.fc1.weight"
            ]
            out_dict["head.pre_logits.fc.bias"] = state_dict["visual.head.mlp.fc1.bias"]
            out_dict["head.fc.weight"] = state_dict["visual.head.mlp.fc2.weight"]
            out_dict["head.fc.bias"] = torch.zeros(
                state_dict["visual.head.mlp.fc2.weight"].shape[0]
            )
        return out_dict

    import re

    for k, v in state_dict.items():
        k = k.replace("downsample_layers.0.", "stem.")
        k = re.sub(r"stages.([0-9]+).([0-9]+)", r"stages.\1.blocks.\2", k)
        k = re.sub(
            r"downsample_layers.([0-9]+).([0-9]+)", r"stages.\1.downsample.\2", k
        )
        k = k.replace("dwconv", "conv_dw")
        k = k.replace("pwconv", "mlp.fc")
        if "grn" in k:
            k = k.replace("grn.beta", "mlp.grn.bias")
            k = k.replace("grn.ramma", "mlp.grn.weight")
            v = v.reshape(v.shape[-1])
        k = k.replace("head.", "head.fc.")
        if k.startswith("norm."):
            k = k.replace("norm", "head.norm")
        if v.ndim == 2 and "head" not in k:
            model_shape = model.state_dict()[k].shape
            v = v.reshape(model_shape)
        out_dict[k] = v

    return out_dict


def _create_convnext(variant, pretrained=False, **kwargs):
    if kwargs.get("pretrained_cfg", "") == "fcmae":
        # NOTE fcmae pretrained weights have no classifier or final norm-layer (`head.norm`)
        # This is workaround loading with num_classes=0 w/o removing norm-layer.
        kwargs.setdefault("pretrained_strict", False)

    model = build_model_with_cfg(
        ConvNeXt,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=(0, 1, 2, 3), flatten_sequential=True),
        **kwargs,
    )
    return model


def convnext_large(pretrained=False, **kwargs) -> ConvNeXt:
    model_args = dict(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536])
    model = _create_convnext(
        "convnext_large", pretrained=pretrained, **dict(model_args, **kwargs)
    )
    return model


class CLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
        self,
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
        output_dict: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.output_dict = output_dict

        self.visual = convnext_large()


class ConvNextVisionEncoder(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.model_type = "convnext_large_d_320"
        self.model_channel = [192, 384, 768, 1536]  # stage 0-3

        clip_model = CLIP(**get_model_config(self.model_type), use_text=False)

        # decompose stem and stages blocks in vision tower
        self.vision_stem = clip_model.visual.stem
        self.vision_stages = clip_model.visual.stages

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
        with torch.no_grad():
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


if __name__ == "__main__":
    model = ConvNextVisionEncoder()
    print(model.state_dict().keys())
