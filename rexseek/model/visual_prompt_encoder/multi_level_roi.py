from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from rexseek.model.utils import gen_sineembed_for_position


class MultiLevelROIVisualPrompt(nn.Module):
    """Initialize the MultiLevelROIVisualPrompt.

    Args:
        output_size (Optional[int]): The size of the output. Default is None.
        channel_per_level (List[int]): List of channels per level. Default is [192, 384, 768, 1536].
        spatial_scale (Optional[float]): The spatial scale factor. Default is None.
        with_additional_projection (bool): Whether to use additional projection. Default is False.
        visual_prompt_hidden_size (int): The hidden size of the visual prompt. Default is 1024.
        add_pos_embedding (bool): Whether to add position embedding. Default is False.
        pos_embedding_dim (int): The dimension of the position embedding. Default is 1024.
    """

    def __init__(
        self,
        output_size: int = 7,
        channel_per_level: List[int] = [192, 384, 768, 1536],
        spatail_scale: float = 192 / 768,
        add_pos_embedding: bool = True,
        pos_embedding_dim: int = 2880,
    ):
        super(MultiLevelROIVisualPrompt, self).__init__()
        self.output_size = output_size

        self.channel_per_level = channel_per_level
        self.spatail_scale = spatail_scale
        self.add_pos_embedding = add_pos_embedding
        self.pos_embedding_dim = pos_embedding_dim

    def __call__(
        self,
        multi_level_features: List[torch.Tensor],
        boxes: Union[torch.Tensor, List[torch.Tensor]],
    ) -> torch.Tensor:
        """Performs Region of Interest (RoI) Align operator on multi-level features. The RoI
        feature on each scale will go through a different linear layer for projection. Different
        RoI features will be summed up and then average pooled.

        Args:
            multi_level_features (Listp[Tensor[N, C, H, W]]): Feature maps from different levels
            boxes (Tensor[K, 5] or List[Tensor[L, 4]]): the box coordinates in (x1, y1, x2, y2)
                format where the regions will be taken from.
        Returns:
            Tensor[1, K, C]: The output tensor that has the shape KxC, where K is the number of RoIs
        """
        ori_dtype = multi_level_features[0].dtype
        boxes[0] = boxes[0].float()
        concat_multi_level_feature = []
        max_height = max([feature.shape[2] for feature in multi_level_features])
        max_width = max([feature.shape[3] for feature in multi_level_features])
        # interpolate to the same size
        for level, feature in enumerate(multi_level_features):
            if level != 0:
                concat_multi_level_feature.append(
                    F.interpolate(
                        feature.float(),
                        size=(max_height, max_width),
                        mode="bilinear",
                        align_corners=False,
                    )
                )
            else:
                concat_multi_level_feature.append(feature.float())
        concat_multi_level_feature = torch.cat(concat_multi_level_feature, dim=1)

        out_box_feat = roi_align(
            concat_multi_level_feature,
            boxes,
            output_size=self.output_size,
            spatial_scale=self.spatail_scale,
        )

        # Average Pooling -> n,c -> 1,n,c
        out_box_feat = out_box_feat.mean(dim=(2, 3)).reshape(
            1, out_box_feat.shape[0], out_box_feat.shape[1]
        )
        if self.add_pos_embedding:
            # note that this boxes is in xyxy, unormalized format, so we need to normalize it first
            boxes = boxes[0]  # (N, 4)
            boxes = boxes.to(out_box_feat.dtype)
            original_img_width = max_width / self.spatail_scale
            original_img_height = max_height / self.spatail_scale
            boxes[:, [0, 2]] = boxes[:, [0, 2]] / original_img_width
            boxes[:, [1, 3]] = boxes[:, [1, 3]] / original_img_height
            # convert from xyxy to cx, cy, w, h
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
            boxes[:, 0] = boxes[:, 0] + boxes[:, 2] / 2
            boxes[:, 1] = boxes[:, 1] + boxes[:, 3] / 2
            pos_embed = gen_sineembed_for_position(
                boxes.unsqueeze(0), self.pos_embedding_dim // 4
            )
            out_box_feat = out_box_feat + pos_embed

        return out_box_feat
