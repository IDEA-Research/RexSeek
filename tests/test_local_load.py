import json
import re

import torch
from PIL import Image

from rexseek.builder import load_rexseek_model
from rexseek.tools import visualize_rexseek_output
from rexseek.utils.inference_utils import (
    modify_processor_resolution,
    prepare_input_for_rexseek,
)

if __name__ == "__main__":
    model_path = "IDEA-Research/RexSeek-3B"
    # load model
    tokenizer, rexseek_model, image_processor, context_len = load_rexseek_model(
        model_path
    )
    image_processor, crop_size_raw = modify_processor_resolution(image_processor)

    # load image
    test_image_path = "tests/images/Cafe.jpg"
    image = Image.open(test_image_path)
    candidate_boxes_path = "tests/images/Cafe_person.json"
    question = (
        "Please detect male in this image. Answer the question with object indexes."
    )

    with open(candidate_boxes_path, "r") as f:
        candidate_boxes = json.load(f)  # boxes in xyxy format
    template = dict(
        SYSTEM=("<|im_start|>system\n{system}<|im_end|>\n"),
        INSTRUCTION=("<|im_start|>user\n{input}<|im_end|>\n" "<|im_start|>assistant\n"),
        SUFFIX="<|im_end|>",
        SUFFIX_AS_EOS=True,
        SEP="\n",
        STOP_WORDS=["<|im_end|>", "<|endoftext|>"],
    )
    data_dict = prepare_input_for_rexseek(
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
