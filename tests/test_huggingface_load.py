import json

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from rexseek.tools import visualize_rexseek_output

if __name__ == "__main__":
    model_path = "IDEA-Research/RexSeek-3B"
    # load the processor
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="cuda",
    )

    print(f"loading RexSeek-3B model...")
    # load rexseek model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_safetensors=True,
    ).to("cuda")
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    # load iamge
    test_image_path = "tests/images/Cafe.jpg"
    image = Image.open(test_image_path)
    candidate_boxes_path = "tests/images/Cafe_person.json"
    with open(candidate_boxes_path, "r") as f:
        candidate_boxes = json.load(f)
    # prepare input
    data_dict = processor.process(
        image=image,
        question="Please detect two person sitting by the table in this image. Answer the question with object indexes.",
        bbox=candidate_boxes,
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
        output_ids = model.generate(
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
