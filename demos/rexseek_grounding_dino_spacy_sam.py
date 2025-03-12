import argparse
import re
from typing import Any, Dict, List, Union

import groundingdino.datasets.transforms as T
import numpy as np
import spacy
import torch
from groundingdino.util.inference import load_model, predict
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

from rexseek.builder import load_rexseek_model
from rexseek.tools import visualize_rexseek_output
from rexseek.utils.inference_utils import (
    modify_processor_resolution,
    prepare_input_for_rexseek,
)

nlp = spacy.load("en_core_web_sm")


def parse_args():
    parser = argparse.ArgumentParser(
        description="RexSeek Demo with Grounding DINO and Spacy"
    )
    parser.add_argument(
        "--image", type=str, default="tests/images/Cafe.jpg", help="path to input image"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tests/images/Cafe_with_answer_gdino.jpeg",
        help="path to output image",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="IDEA-Research/RexSeek-3B",
        help="path to RexSeek model",
    )
    parser.add_argument(
        "--gdino-config",
        type=str,
        default="demos/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        help="path to Grounding DINO config",
    )
    parser.add_argument(
        "--gdino-weights",
        type=str,
        default="demos/GroundingDINO/weights/groundingdino_swint_ogc.pth",
        help="path to Grounding DINO weights",
    )
    parser.add_argument(
        "--sam-weights",
        type=str,
        default="demos/segment-anything/weights/sam_vit_h_4b8939.pth",
        help="path to SAM weights",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.25,
        help="text threshold for Grounding DINO",
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=0.35,
        help="box threshold for Grounding DINO",
    )
    parser.add_argument(
        "--referring",
        type=str,
        default="male",
        help="referring to detect in the image",
    )
    return parser.parse_args()


def convert_all_cate_prediction_to_ans(prediction: str) -> Dict[str, List[str]]:
    # Define the pattern to extract ground truth labels and object tags
    pattern = r"<ground>(.*?)<\/ground><objects>(.*?)<\/objects>"

    # Find all matches in the prediction string
    matches = re.findall(pattern, prediction)

    # Initialize the dictionary to store the result
    ans = {}
    existed_obj_index = []
    for ground, objects in matches:
        ground = ground.strip()
        # Extract the object tags, e.g., <obj0>, <obj1>, <obj2>
        object_tags = re.findall(r"<obj\d+>", objects)
        filtered_object_tags = []
        for object_tag in object_tags:
            if object_tag in existed_obj_index:
                continue
            filtered_object_tags.append(object_tag)
            existed_obj_index.append(object_tag)

        if len(filtered_object_tags) == 0:
            continue
        # Add the ground truth label and associated object tags to the dictionary
        if ground not in ans:
            ans[ground] = object_tags

    return ans


def spacy_noun_phrases(caption):
    """Extract noun phrases from a given text caption using spaCy.

    Args:
        caption (str): The input text caption to process.

    Returns:
        list[str]: A list of lowercase noun phrases extracted from the caption.
    """
    doc = nlp(caption)
    nouns = []
    for nounc in doc.noun_chunks:
        nouns.append(str(nounc).lower())
    return nouns


def gdino_load_image(image: Union[str, Image.Image]):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    if isinstance(image, str):
        image_source = Image.open(image).convert("RGB")
    else:
        image_source = image
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image_transformed


def inference_rexseek(
    image: Image.Image,
    image_processor: Any,
    tokenizer: Any,
    rexseek_model: Any,
    crop_size_raw: tuple[int, int],
    candidate_boxes: torch.Tensor,
    question: str,
) -> str:
    """Process an image with RexSeek model to answer questions about detected objects.

    Args:
        image: Input PIL image
        image_processor: Processor for image preprocessing
        tokenizer: Tokenizer for text processing
        rexseek_model: The RexSeek model instance
        crop_size_raw: Tuple of (height, width) for image cropping
        template: Dictionary containing prompt templates for model input
        candidate_boxes: Tensor of detected object bounding boxes
        question: Question to ask about the detected objects

    Returns:
        str: Model's answer to the question about detected objects
    """
    # start inference for rexseek
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
            temperature=0.0,
            top_p=None,
        )
    answer = tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0].strip()
    return answer


def inference_gdino(
    image: Image.Image,
    prompts: list[str],
    gdino_model: Any,
    TEXT_TRESHOLD=0.25,
    BOX_TRESHOLD=0.35,
) -> torch.Tensor:
    """Process an image with Grounding DINO model to detect objects.

    Args:
        image: Input PIL image
        gdino_model: The Grounding DINO model instance
        gdino_processor: Processor for Grounding DINO input preparation

    Returns:
        torch.Tensor: Tensor containing detected object bounding boxes in shape (N, 4)
    """
    text_labels = ".".join(prompts)
    image_transformed = gdino_load_image(image)
    boxes, _, _ = predict(
        model=gdino_model,
        image=image_transformed,
        caption=text_labels,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD,
    )
    # the output boxes is in the format of (x,y,w,h), in [0,1]
    boxes = boxes * torch.tensor([image.width, image.height, image.width, image.height])
    # convert to the format of (x1,y1,x2,y2)
    boxes = torch.cat(
        (boxes[:, :2] - boxes[:, 2:4] / 2, boxes[:, :2] + boxes[:, 2:4] / 2), dim=1
    )
    return boxes


def inference_sam(
    image: Image.Image, sam_predictor: Any, boxes: torch.Tensor
) -> np.ndarray:
    """Process an image with Segment Anything Model (SAM) to generate masks for detected objects.

    Args:
        image: Input PIL image to generate masks for
        sam_predictor: SAM model predictor instance
        boxes: Tensor of detected object bounding boxes in format (N, 4)

    Returns:
        np.ndarray: Array of binary masks for each detected object, shape (N, H, W)
    """
    boxes = boxes.to("cuda")
    img = np.array(image)
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes, img.shape[:2])
    sam_predictor.set_image(img, image_format="BGR")
    with torch.no_grad():
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
    masks = masks.cpu().numpy()  # (Num_mask, 1 ,H, W)
    return masks


if __name__ == "__main__":
    args = parse_args()
    referring = args.referring
    sub_objects = spacy_noun_phrases(referring)
    # Start loading models
    # load gdino model
    gdino_model = load_model(
        args.gdino_config,
        args.gdino_weights,
    ).to("cuda")

    # load rexseek model
    tokenizer, rexseek_model, image_processor, context_len = load_rexseek_model(
        args.model_path
    )
    image_processor, crop_size_raw = modify_processor_resolution(image_processor)

    # load sam model
    sam = sam_model_registry["vit_h"](checkpoint=args.sam_weights)
    sam.to("cuda")
    sam_predictor = SamPredictor(sam)

    # load image
    test_image_path = args.image
    image = Image.open(test_image_path)

    # inference gdino
    candidate_boxes = inference_gdino(
        image,
        sub_objects,
        gdino_model,
        TEXT_TRESHOLD=args.text_threshold,
        BOX_TRESHOLD=args.box_threshold,
    )
    print(f"Detected boxes: {candidate_boxes}")

    # inference rexseek
    question = f"Please detect {args.referring} in this image. Answer the question with object indexes."
    answer = inference_rexseek(
        image,
        image_processor,
        tokenizer,
        rexseek_model,
        crop_size_raw,
        candidate_boxes,
        question,
    )
    print(f"RexSeek answer: {answer}")
    ans = convert_all_cate_prediction_to_ans(answer)
    pred_boxes = []
    for k, v in ans.items():
        for box in v:
            obj_idx = int(box[4:-1])
            if obj_idx < len(candidate_boxes):
                pred_boxes.append(candidate_boxes[obj_idx])

    # inference sam
    masks = inference_sam(image, sam_predictor, torch.cat(pred_boxes, 0))
    image_with_boxes = visualize_rexseek_output(
        image,
        input_boxes=candidate_boxes,
        masks=masks,
        prediction_text=answer,
    )
    image_with_boxes.save(args.output)
    print(f"Results saved to: {args.output}")
