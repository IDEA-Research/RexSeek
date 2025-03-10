import json
import re

import torch
import torchvision.transforms.functional as F
from PIL import Image

from rexseek.builder import load_rexseek_model
from rexseek.tools import visualize_rexseek_output
from rexseek.utils import xywh_to_xyxy, xyxy_to_xywh
from rexseek.utils.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_OBJECT_FEATURE_TOKEN,
    DEFAULT_OBJECT_TOKEN,
)
from rexseek.utils.inference_utils import (
    expand2square,
    modify_processor_resolution,
    pad_boxes,
    resize_boxes,
    tokenizer_image_object_token,
)


def prepare_input(
    image, image_processor, tokenizer, bbox, question: str, crop_size_raw, template
):
    """Prepare input data for inference.

    Args:
        image (Union[str, Image.Image]): The image to process.
        bbox (List[List[int]]): A list of bounding boxes for the image. Each bounding box should
            be in order of [x, y,  x, y].
        question (str): The question to ask about the image.
    """
    data_dict = {}
    # step1 load image
    if type(image) == str:
        image = Image.open(image).convert("RGB")
    ori_w, ori_h = F.get_image_size(image)
    image = expand2square(
        image,
        tuple(int(x * 255) for x in image_processor.image_mean),
    )
    pad_w, pad_h = F.get_image_size(image)
    image_aux = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][
        0
    ]
    resize_h, resize_w = image_aux.shape[-2:]
    data_dict["pixel_values_aux"] = image_aux.unsqueeze(0)
    image = image_aux.clone()
    image = torch.nn.functional.interpolate(
        image[None],
        size=[crop_size_raw["height"], crop_size_raw["width"]],
        mode="bilinear",
        align_corners=False,
    )[0]
    data_dict["pixel_values"] = image.unsqueeze(0)

    # step2 load boxes
    bbox = xyxy_to_xywh(bbox)
    bbox = pad_boxes(bbox, (ori_w, ori_h))
    bbox = resize_boxes(bbox, (pad_w, pad_h), (resize_h, resize_w))
    data_dict["gt_boxes"] = torch.tensor(xywh_to_xyxy(bbox)).unsqueeze(0)

    # step3 prepare question
    total_num_boxes = len(bbox)
    obj_tokens = [
        DEFAULT_OBJECT_TOKEN.replace("<i>", str(i)) for i in range(total_num_boxes)
    ]
    obj_tokens = (
        DEFAULT_OBJECT_FEATURE_TOKEN.join(obj_tokens) + DEFAULT_OBJECT_FEATURE_TOKEN
    )
    question = question.replace(DEFAULT_IMAGE_TOKEN, "")
    question = DEFAULT_IMAGE_TOKEN + "\n" + obj_tokens + "\n" + question

    inputs = ""
    inputs += template["INSTRUCTION"].format(input=question, round=1)

    # step4 tokenize question
    input_ids = tokenizer_image_object_token(inputs, tokenizer)
    data_dict["input_ids"] = torch.tensor(input_ids).unsqueeze(0)

    return data_dict


if __name__ == "__main__":
    model_path = "RexSeek-3B"
    # load model
    tokenizer, rexseek_model, image_processor, context_len = load_rexseek_model(
        model_path
    )
    img_size_clip = 336
    image_size_aux = 768
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
    image_processor = modify_processor_resolution(image_processor)

    # load iamge
    test_image_path = "tests/images/Cafe.jpg"
    image = Image.open(test_image_path)
    candidate_boxes_path = "tests/images/Cafe_person.json"
    question = (
        "Please detect male in this image. Answer the question with object indexes."
    )

    with open(candidate_boxes_path, "r") as f:
        candidate_boxes = json.load(f)
    template = dict(
        SYSTEM=("<|im_start|>system\n{system}<|im_end|>\n"),
        INSTRUCTION=("<|im_start|>user\n{input}<|im_end|>\n" "<|im_start|>assistant\n"),
        SUFFIX="<|im_end|>",
        SUFFIX_AS_EOS=True,
        SEP="\n",
        STOP_WORDS=["<|im_end|>", "<|endoftext|>"],
    )
    data_dict = prepare_input(
        image,
        image_processor,
        tokenizer,
        candidate_boxes,
        question,
        crop_size_raw,
        template,
    )

    input_ids = data_dict["input_ids"]
    pixel_values = data_dict["pixel_values"]
    pixel_values_aux = data_dict["pixel_values_aux"]
    gt_boxes = data_dict["gt_boxes"]

    input_ids = input_ids.to(device="cuda", non_blocking=True)
    pixel_values = pixel_values.to(
        device="cuda", dtype=torch.float16, non_blocking=True
    )
    pixel_values_aux = pixel_values_aux.to(
        device="cuda", dtype=torch.float16, non_blocking=True
    )

    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        output_ids = rexseek_model.generate(
            input_ids,
            pixel_values=pixel_values,
            pixel_values_aux=pixel_values_aux,
            gt_boxes=gt_boxes.to(dtype=torch.float16, device="cuda"),
            do_sample=False,
            max_new_tokens=512,
        )
    answer = tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0].strip()
    print(f"answer: {answer}")
    image_with_boxes = visualize_rexseek_output(
        image,
        input_boxes=candidate_boxes,
        prediction_text=answer,
    )
    image_with_boxes.save("tests/images/Cafe_with_answer.jpeg")
