import re

import numpy as np
from PIL import Image

from rexseek.utils.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_OBJECT_FEATURE_TOKEN,
    DEFAULT_OBJECT_INDEX,
    IMAGE_TOKEN_INDEX,
)


def split_special_strings(input_string: str, special_strings: list[str] = None):
    """Split the input string into a list of strings, keeping the special strings.

    Args:
        input_string (str): The input string to split.

        Example:

            input_string = "<image>\n<obj0><objfeat><obj1><objfeat>\n I am happy today."
            output = ['<image>', '\n<obj0>', '<objfeat>', '<obj1>', '<objfeat>', '\n I am happy today.']

    Returns:
        list: A list of strings, with the special strings separated from the rest of the input string.
    """
    # Create a regex pattern to match the special strings
    pattern = "|".join(map(re.escape, special_strings))

    # Split the input string using the pattern, keeping the special strings in the result
    split_list = re.split(f"({pattern})", input_string)

    # Remove empty strings from the list
    split_list = [s for s in split_list if s]

    return split_list


def modify_processor_resolution(image_processor, img_size_clip=336, image_size_aux=768):
    if hasattr(image_processor, "crop_size"):
        if img_size_clip is None:
            crop_size_raw = image_processor.crop_size.copy()
        else:
            crop_size_raw = dict(height=img_size_clip, width=img_size_clip)
        image_processor.crop_size["height"] = image_size_aux
        image_processor.crop_size["width"] = image_size_aux
        image_processor.size["shortest_edge"] = image_size_aux
        is_clip = True
    else:
        if img_size_clip is None:
            crop_size_raw = image_processor.crop_size.copy()
        else:
            crop_size_raw = dict(height=img_size_clip, width=img_size_clip)
        image_processor.size["height"] = image_size_aux
        image_processor.size["width"] = image_size_aux
    return image_processor


def pad_boxes(gt_boxes, old_size):
    old_w, old_h = old_size
    gt_boxes = np.array(gt_boxes).astype(np.float32)
    # Calculate the padding added
    if old_w > old_h:
        pad_top = (old_w - old_h) // 2
        pad_bottom = old_w - old_h - pad_top
        pad_left, pad_right = 0, 0
    else:
        pad_left = (old_h - old_w) // 2
        pad_right = old_h - old_w - pad_left
        pad_top, pad_bottom = 0, 0

    # Adjust the boxes for padding
    gt_boxes[:, 0] += pad_left  # x
    gt_boxes[:, 1] += pad_top  # y
    return gt_boxes


def resize_boxes(gt_boxes, old_size, new_size):
    old_w, old_h = old_size
    new_h, new_w = new_size
    gt_boxes = np.array(gt_boxes).astype(np.float32)
    # Calculate scale factors
    scale_x = new_w / max(old_w, old_h)
    scale_y = new_h / max(old_w, old_h)

    # Resize the boxes
    gt_boxes[:, 0] *= scale_x  # x
    gt_boxes[:, 1] *= scale_y  # y
    gt_boxes[:, 2] *= scale_x  # w
    gt_boxes[:, 3] *= scale_y  # h

    return gt_boxes


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def tokenizer_image_object_token(prompt, tokenizer):
    bos_token_id = tokenizer.bos_token_id
    split_tokens = [DEFAULT_IMAGE_TOKEN, DEFAULT_OBJECT_FEATURE_TOKEN]
    chunks = split_special_strings(prompt, split_tokens)
    input_encode = [bos_token_id] if bos_token_id is not None else []
    for chunk in chunks:
        if chunk == DEFAULT_IMAGE_TOKEN:
            input_encode.append(IMAGE_TOKEN_INDEX)
        elif chunk == DEFAULT_OBJECT_FEATURE_TOKEN:
            input_encode.append(DEFAULT_OBJECT_INDEX)
        else:
            input_encode.extend(tokenizer.encode(chunk, add_special_tokens=False))
    return input_encode
