import numpy as np


def xywh_to_xyxy(boxes):
    """
    Convert boxes from xywh to xyxy format.

    Parameters:
    boxes (numpy.ndarray): An array of shape (N, 4) where N is the number of boxes.
                           Each box is represented as [x, y, width, height].

    Returns:
    numpy.ndarray: An array of shape (N, 4) where each box is represented as [x_min, y_min, x_max, y_max].
    """
    boxes = np.array(boxes)
    x, y, width, height = (
        boxes[:, 0],
        boxes[:, 1],
        boxes[:, 2],
        boxes[:, 3],
    )
    x_max = x + width
    y_max = y + height
    return np.stack([x, y, x_max, y_max], axis=1)


def xyxy_to_xywh(boxes):
    """
    Convert boxes from xywh to xyxy format.

    Parameters:
    boxes (numpy.ndarray): An array of shape (N, 4) where N is the number of boxes.
                           Each box is represented as [x, y, x, y].

    Returns:
    numpy.ndarray: An array of shape (N, 4) where each box is represented as [x_min, y_min, w, h].
    """
    boxes = np.array(boxes)
    x_min, y_min, x_max, y_max = (
        boxes[:, 0],
        boxes[:, 1],
        boxes[:, 2],
        boxes[:, 3],
    )
    w = x_max - x_min
    h = y_max - y_min
    return np.stack([x_min, y_min, w, h], axis=1)


def to_cxcywh_normailize(boxes, image_w_h):
    """Convert the boxes from xyxy to cxcywh format and normalize the values.

    Args:
        boxes (numpy.ndarray): An array of shape (N, 4) where N is the number of boxes.
                               Each box is represented as [x_min, y_min, x_max, y_max].
        image_w_h (tuple): The width and height of the image.
    """
    x_min, y_min, x_max, y_max = (
        boxes[:, 0],
        boxes[:, 1],
        boxes[:, 2],
        boxes[:, 3],
    )
    x_c = (x_min + x_max) / 2 / image_w_h[0]
    y_c = (y_min + y_max) / 2 / image_w_h[1]
    w = (x_max - x_min) / image_w_h[0]
    h = (y_max - y_min) / image_w_h[1]
    # cut the value between 0 and 1
    x_c = np.clip(x_c, 0, 1)
    y_c = np.clip(y_c, 0, 1)
    w = np.clip(w, 0, 1)
    h = np.clip(h, 0, 1)
    return np.stack([x_c, y_c, w, h], axis=1)
