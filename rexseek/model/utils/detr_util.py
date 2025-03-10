from typing import Union

import torch
from torchvision.ops.boxes import box_area


def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def gen_encoder_output_proposals(
    memory: torch.Tensor,
    memory_padding_mask: torch.Tensor,
    spatial_shapes: torch.Tensor,
    learnedwh=None,
):
    """
    Input:
        - memory: bs, \sum{hw}, d_model
        - memory_padding_mask: bs, \sum{hw}
        - spatial_shapes: nlevel, 2
        - learnedwh: 2
    Output:
        - output_memory: bs, \sum{hw}, d_model
        - output_proposals: bs, \sum{hw}, 4
    """
    N_, S_, C_ = memory.shape
    base_scale = 4.0
    proposals = []
    _cur = 0
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        mask_flatten_ = memory_padding_mask[:, _cur : (_cur + H_ * W_)].view(
            N_, H_, W_, 1
        )
        valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
        valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device),
        )
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)  # H_, W_, 2

        scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(
            N_, 1, 1, 2
        )
        grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale

        if learnedwh is not None:
            wh = torch.ones_like(grid) * learnedwh.sigmoid() * (2.0**lvl)
        else:
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)

        proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
        proposals.append(proposal)
        _cur += H_ * W_

    output_proposals = torch.cat(proposals, 1)
    output_proposals_valid = (
        (output_proposals > 0.01) & (output_proposals < 0.99)
    ).all(-1, keepdim=True)
    output_proposals = torch.log(output_proposals / (1 - output_proposals))  # unsigmoid
    output_proposals = output_proposals.masked_fill(
        memory_padding_mask.unsqueeze(-1), float("inf")
    )
    output_proposals = output_proposals.masked_fill(
        ~output_proposals_valid, float("inf")
    )

    output_memory = memory
    output_memory = output_memory.masked_fill(
        memory_padding_mask.unsqueeze(-1), float(0)
    )
    output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))

    return output_memory, output_proposals


# modified from torchvision to also return the union
def box_iou(
    boxes1: torch.Tensor, boxes2: torch.Tensor
) -> Union[torch.Tensor, torch.Tensor]:
    """
    Compute the intersection over union (IoU) between two sets of bounding boxes.

    Args:
        boxes1 (Tensor): Bounding boxes in format (x1, y1, x2, y2). Shape (N, 4).
        boxes2 (Tensor): Bounding boxes in format (x1, y1, x2, y2). Shape (M, 4).

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing two tensors:
            - iou (Tensor): The IoU between the two sets of bounding boxes. Shape (N, M).
            - union (Tensor): The area of the union between the two sets of bounding boxes. Shape (N, M).
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # import ipdb; ipdb.set_trace()
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)
    return iou, union, inter


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute the Generalized Intersection over Union (IoU) between two sets of bounding boxes.

    The boxes should be in [x0, y0, x1, y1] format.

    Args:
        boxes1 (Tensor): A tensor of shape (N, 4) containing the bounding boxes for the first set.
        boxes2 (Tensor): A tensor of shape (M, 4) containing the bounding boxes for the second set.

    Returns:
        Tensor: A tensor of shape (N, M) containing the pairwise Generalized IoU between the
            two sets of bounding boxes.
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    # except:
    #     import ipdb; ipdb.set_trace()
    iou, union, inter = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / (area + 1e-6)


# return inter
