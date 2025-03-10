import math

import torch


def gen_sineembed_for_position(pos_tensor, dim_of_pos_feats):
    """Generate sine position embedding from a position tensor.

    Args:
        pos_tensor (torch.Tensor): shape: [batch_size, N, 4]. the last dimension is [cx, cy, w, h] in
            normalized coordinates in range [0, 1].
        out_dim (int): the output dimension of the position embedding.

    Returns:
        pos (torch.Tensor): shape: [batch_size, N, out_dim].
    """
    scale = 2 * math.pi
    dim_t = torch.arange(
        dim_of_pos_feats, dtype=torch.float32, device=pos_tensor.device
    )
    dim_t = 10000 ** (2 * (dim_t // 2) / dim_of_pos_feats)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack(
        (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
    ).flatten(2)
    pos_y = torch.stack(
        (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
    ).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack(
            (pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3
        ).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack(
            (pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3
        ).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos
