from typing import List, Tuple

import torch
from torchvision.ops import roi_align


def extract_roi_from_single_feature_map(
    boxes: torch.Tensor,
    image_feature: torch.Tensor,
    roi_size: int,
    image_size: Tuple[int, int],
    downsample_ratio: int = 14,
):
    """Extract roi aligned feature from image feature map. we will compute the average of the
    feature map in the roi

    Args:
        boxes: [num_boxes, 4] (x, y, x, y). This is normalized coordinates
        image_feature: [N, C]. This is flattened feature map
        roi_size: int. The size of the roi
        image_size: [H, W]. The size of the image
        downsample_ratio: int. The downsample ratio of the image feature map

    Returns:
        rois: [num_boxes, C]
    """
    height, width = image_size
    boxes[:, [0, 2]] *= width  # x and x
    boxes[:, [1, 3]] *= height  # y and y

    # Add batch indices to boxes (assuming all boxes are in the same image, batch index is 0)
    ori_dtype = boxes.dtype
    batch_indices = torch.zeros(
        (boxes.size(0), 1), device=boxes.device, dtype=boxes.dtype
    )
    rois = torch.cat([batch_indices, boxes], dim=1)

    # Calculate the spatial size of the feature map
    feature_map_height = height // downsample_ratio
    feature_map_width = width // downsample_ratio
    C = image_feature.size(1)

    # Adjust the feature map size to match the expected input of roi_align
    image_feature = image_feature.transpose(1, 0).view(
        1, C, feature_map_height, feature_map_width
    )

    # Set the output size (e.g., pooled height and width)
    output_size = (roi_size, roi_size)  # This can be changed as per the requirement

    # Extract ROI-aligned features
    roi_aligned_features = roi_align(
        image_feature.to(torch.float32),
        rois.to(torch.float32),
        output_size,
        spatial_scale=1.0 / downsample_ratio,
    )

    # average the last two
    roi_aligned_features = roi_aligned_features.mean(dim=[2, 3])

    return roi_aligned_features.to(ori_dtype)
