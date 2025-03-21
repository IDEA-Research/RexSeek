import argparse
import json
import math
import os
import re

import numpy as np
import shortuuid
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from rexseek.builder import load_rexseek_model
from rexseek.utils import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_OBJECT_FEATURE_TOKEN,
    DEFAULT_OBJECT_INDEX,
    DEFAULT_OBJECT_TOKEN,
    IMAGE_TOKEN_INDEX,
    to_cxcywh_normailize,
    xywh_to_xyxy,
)
from rexseek.utils.inference_utils import expand2square


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="IDEA-Research/RexSeek-3B",
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        default="IDEA-Research/HumanRef/images",
    )
    parser.add_argument(
        "--question_file",
        type=str,
        default="IDEA-Research/HumanRef/annotations.jsonl",
    )
    parser.add_argument(
        "--answers_file",
        type=str,
        default="IDEA-Research/HumanRef/evaluation_results/eval_rexseek/RexSeek-3B_results.jsonl",
    )
    parser.add_argument("--template", type=str, default="qwen_chat")
    parser.add_argument(
        "--question-template",
        type=str,
        default="Please detect [OBJ] in this image. Answer the question with object indexes.",
    )
    parser.add_argument(
        "--use_add_boxes",
        type=str,
        default="no",
    )
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    args = parser.parse_args()
    return args


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


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


class RexSeekEvalDataset(Dataset):
    """This is for the evaluation of Det like datasets for ChatRex, which we will append object
    queryes in the annotation to the question and convert the answer.

    """

    def __init__(
        self,
        data_list,
        image_folder,
        template: dict,
        tokenizer,
        image_processor,
        image_size_aux=768,
        img_size_clip=336,
        need_return_score=False,
        box2cxcywh_normailize=False,
        use_system=False,
        metainfo=dict(name="gqa"),
    ):
        self.image_folder = image_folder
        self.data_list = data_list
        self.template = template
        self.metainfo = metainfo
        self.image_size_aux = image_size_aux
        self.is_clip = False
        self.use_system = use_system
        self.need_return_score = need_return_score
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        if hasattr(self.image_processor, "crop_size"):
            if img_size_clip is None:
                self.crop_size_raw = self.image_processor.crop_size.copy()
            else:
                self.crop_size_raw = dict(height=img_size_clip, width=img_size_clip)
            self.image_processor.crop_size["height"] = image_size_aux
            self.image_processor.crop_size["width"] = image_size_aux
            self.image_processor.size["shortest_edge"] = image_size_aux
            self.is_clip = True
        else:
            if img_size_clip is None:
                self.crop_size_raw = self.image_processor.crop_size.copy()
            else:
                self.crop_size_raw = dict(height=img_size_clip, width=img_size_clip)
            self.image_processor.size["height"] = image_size_aux
            self.image_processor.size["width"] = image_size_aux
        self.box2cxcywh_normailize = box2cxcywh_normailize

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        data_dict = {"img_id": data["img_id"]}
        # step 1 load and process image
        image_path = os.path.join(self.image_folder, data["image_path"])
        if not os.path.exists(image_path):
            image_path = image_path.replace("val2017", "train2017")
        image = Image.open(image_path).convert("RGB")

        ori_w, ori_h = F.get_image_size(image)

        image = expand2square(
            image,
            tuple(int(x * 255) for x in self.image_processor.image_mean),
        )
        pad_w, pad_h = F.get_image_size(image)

        image_aux = self.image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]
        resize_h, resize_w = image_aux.shape[-2:]
        data_dict["pixel_values_aux"] = image_aux

        image = image_aux.clone()
        image = torch.nn.functional.interpolate(
            image[None],
            size=[self.crop_size_raw["height"], self.crop_size_raw["width"]],
            mode="bilinear",
            align_corners=False,
        )[0]
        data_dict["pixel_values"] = image
        # step 2 load boxes
        if "gt_boxes" in data and len(data["gt_boxes"]) > 0:
            gt_boxes = self.pad_boxes(data["gt_boxes"], (ori_w, ori_h))
            gt_boxes = self.resize_boxes(gt_boxes, (pad_w, pad_h), (resize_h, resize_w))
            data_dict["gt_boxes"] = torch.tensor(xywh_to_xyxy(gt_boxes))

            # step 3 load question
            total_num_boxes = len(gt_boxes)
            obj_tokens = [
                DEFAULT_OBJECT_TOKEN.replace("<i>", str(i))
                for i in range(total_num_boxes)
            ]
            obj_tokens = (
                DEFAULT_OBJECT_FEATURE_TOKEN.join(obj_tokens)
                + DEFAULT_OBJECT_FEATURE_TOKEN
            )
        else:
            obj_tokens = ""
            data_dict["gt_boxes"] = []

        # step 4 load question
        if self.metainfo["name"] == "multiple_choice":
            # MultipleChoiceDataset
            data_dict["index"] = data["index"]
            if data["context"] is not None:
                text = (
                    data["context"] + "\n" + data["question"] + "\n" + data["options"]
                )
            else:
                text = data["question"] + "\n" + data["options"]
            text = text.replace("<image>\n", "")
            text = DEFAULT_IMAGE_TOKEN + "\n" + obj_tokens + "\n" + text
            text = text + (
                "Answer with the option's letter from the given choices directly."
            )
        elif self.metainfo["name"] in ["chartqa", "gvqa"]:
            # TODO prompt are different of vlmevalkit
            text = (
                data["question"]
                + "\nAnswer the question using a single word or phrase."
            )
            text = text.replace("<image>\n", "")
            text = DEFAULT_IMAGE_TOKEN + "\n" + obj_tokens + "\n" + text
        elif self.metainfo["name"] == "tallyqa":
            text = data["question"]
            text = text.replace("<image>\n", "")
            text = text + "\nAnswer the question using a single number."
            text = DEFAULT_IMAGE_TOKEN + "\n" + obj_tokens + "\n" + text
        elif self.metainfo["name"] in ["hallusion", "pope"]:
            # TODO prompt are different of vlmevalkit
            text = data["question"] + "\nPlease answer the question with yes or no."
            text = text.replace("<image>\n", "")
            text = DEFAULT_IMAGE_TOKEN + "\n" + obj_tokens + "\n" + text
        else:
            text = data["question"]
            if self.metainfo["name"] == "mme":
                text = data["question"].replace(
                    "Please answer yes or no.",
                    "Please answer the question only a single word yes or no.",
                )
            text = text.replace("<image>\n", "")
            text = DEFAULT_IMAGE_TOKEN + "\n" + obj_tokens + "\n" + text

        if self.use_system:
            inputs = self.template.get("SYSTEM", "{system}").format(system="")
        else:
            inputs = ""
        inputs += self.template["INSTRUCTION"].format(input=text, round=1)

        # 3 tokenize inputs
        input_ids = tokenizer_image_object_token(inputs, self.tokenizer)
        data_dict["input_ids"] = torch.tensor(input_ids)
        if self.need_return_score:
            data_dict["need_return_score"] = True

        if self.box2cxcywh_normailize:
            data_dict["gt_boxes"] = to_cxcywh_normailize(
                data_dict["gt_boxes"], (resize_w, resize_h)
            )

        return (
            data_dict["input_ids"],
            data_dict["pixel_values"],
            data_dict["pixel_values_aux"],
            data_dict["gt_boxes"],
        )

    def pad_boxes(self, gt_boxes, old_size):
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

    def resize_boxes(self, gt_boxes, old_size, new_size):
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


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def collate_fn(batch):
    input_ids, pixel_values, pixel_values_aux, gt_boxes = zip(*batch)
    input_ids = torch.stack(input_ids)
    pixel_values = torch.stack(pixel_values)
    pixel_values_aux = torch.stack(pixel_values_aux)
    gt_boxes = torch.stack(gt_boxes)
    return input_ids, pixel_values, pixel_values_aux, gt_boxes


# DataLoader
def create_data_loader(
    question_file,
    image_folder,
    template,
    tokenizer,
    image_processor,
    batch_size=1,
    num_workers=4,
):
    assert batch_size == 1, "batch_size must be 1"
    template = dict(
        SYSTEM=("<|im_start|>system\n{system}<|im_end|>\n"),
        INSTRUCTION=("<|im_start|>user\n{input}<|im_end|>\n" "<|im_start|>assistant\n"),
        SUFFIX="<|im_end|>",
        SUFFIX_AS_EOS=True,
        SEP="\n",
        STOP_WORDS=["<|im_end|>", "<|endoftext|>"],
    )
    dataset = RexSeekEvalDataset(
        question_file,
        image_folder,
        template,
        tokenizer,
        image_processor,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return data_loader


def load_data_list(question_file, question_template):
    question_data = [
        json.loads(q) for q in open(os.path.expanduser(question_file), "r")
    ]
    data_list = []
    for idx in range(len(question_data)):
        sample = question_data[idx]
        index = sample["id"]
        image_path = sample["image_name"]
        referring = sample["referring"]
        question = question_template.replace("[OBJ]", referring)
        gt_boxes = sample["candidate_boxes"]
        add_boxes = (
            sample["add_boxes"] if "add_boxes" in sample else []
        )  # xywh format already
        if "gt_scores" in sample:
            gt_scores = [float(score) for score in sample["gt_scores"]]
        else:
            gt_scores = [1.0] * len(gt_boxes)
        if len(gt_boxes) > 100:
            gt_boxes = gt_boxes[:100]
        # convert gt_boxes from xyxy to xywh
        for i in range(len(gt_boxes)):
            gt_boxes[i] = [
                gt_boxes[i][0],
                gt_boxes[i][1],
                gt_boxes[i][2] - gt_boxes[i][0],
                gt_boxes[i][3] - gt_boxes[i][1],
            ]
        ans = sample["answer_boxes"]

        data = {
            "id": index,
            "img_id": index,
            "image_path": image_path,
            "question": question,
            "gt_boxes": gt_boxes,
            "gt_scores": gt_scores,
            "add_boxes": add_boxes,
            "ans": ans,
        }
        data_list.append(data)
        idx += 1

    return data_list


def convert_all_cate_prediction_to_ans(prediction: str):
    # Define the pattern to extract ground truth labels and object tags
    pattern1 = r"<ground>(.*?)<\/ground><objects>(.*?)<\/objects>"

    # Find all matches in the prediction string
    matches = re.findall(pattern1, prediction)

    if len(matches) == 0:
        pattern1 = r"<ground> (.*?) <\/ground> <objects> (.*?) <\/objects>"
        matches = re.findall(pattern1, prediction)

    # Initialize the dictionary to store the result
    ans = {}

    for ground, objects in matches:
        ground = ground.strip()
        # Extract the object tags, e.g., <obj0>, <obj1>, <obj2>
        object_tags = re.findall(r"<obj\d+>", objects)
        # Add the ground truth label and associated object tags to the dictionary
        if ground not in ans:
            ans[ground] = object_tags

    return ans


def eval_model(args):
    # Model
    disable_torch_init()
    tokenizer, model, image_processor, context_len = load_rexseek_model(
        args.model_path, delay_load=False
    )

    # load data
    data_list = load_data_list(args.question_file, args.question_template)
    data_list = get_chunk(data_list, args.num_chunks, args.chunk_idx)

    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)
    ans_file = open(args.answers_file, "w")

    data_loader = create_data_loader(
        data_list,
        args.image_folder,
        args.template,
        tokenizer,
        image_processor,
    )

    for (input_ids, pixel_values, pixel_values_aux, gt_boxes), line in tqdm(
        zip(data_loader, data_list), total=len(data_list)
    ):
        idx = line["question_id"] if "question_id" in line else line["id"]
        input_ids = input_ids.to(device="cuda", non_blocking=True)
        pixel_values = pixel_values.to(
            device="cuda", dtype=torch.float16, non_blocking=True
        )
        pixel_values_aux = pixel_values_aux.to(
            device="cuda", dtype=torch.float16, non_blocking=True
        )
        if gt_boxes is None:
            gt_boxes = torch.tensor([[]])
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                pixel_values=pixel_values,
                pixel_values_aux=pixel_values_aux,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                gt_boxes=gt_boxes.to(dtype=torch.float16, device="cuda"),
            )
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=False)[
            0
        ].strip()
        converted_dict = convert_all_cate_prediction_to_ans(outputs)
        try:
            pred_indices = list(converted_dict.values())[0]
            pred_indices = [int(re.findall(r"\d+", idx)[0]) for idx in pred_indices]
            pred_boxes = [line["gt_boxes"][idx] for idx in pred_indices]
            # convert from xywh to xyxy
            for i in range(len(pred_boxes)):
                pred_boxes[i] = [
                    pred_boxes[i][0],
                    pred_boxes[i][1],
                    pred_boxes[i][0] + pred_boxes[i][2],
                    pred_boxes[i][1] + pred_boxes[i][3],
                ]
        except:
            pred_boxes = [[]]
        if "There is no such thing in this image" in outputs:
            pred_boxes = [[]]
        ans_id = shortuuid.uuid()
        ans_file.write(
            json.dumps(
                {
                    "id": idx,
                    "question": line["question"],
                    "raw_response": outputs,
                    "extracted_predictions": pred_boxes,
                    "answer_id": ans_id,
                    "candidate_boxes": line["gt_boxes"] if "gt_boxes" in line else None,
                }
            )
            + "\n"
        )
        # ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    args = get_args()
    eval_model(args)
