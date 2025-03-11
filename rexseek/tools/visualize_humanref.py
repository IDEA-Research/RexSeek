import argparse
import concurrent.futures
import json
import os
import random

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as coco_mask
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--anno_path",
        type=str,
        default="IDEA-Research/HumanRef/annotations.jsonl",
    )
    parser.add_argument(
        "--image_root_dir",
        type=str,
        default="IDEA-Research/HumanRef/images",
    )
    parser.add_argument(
        "--domain_anme",
        type=str,
        default="attribute",
    )
    parser.add_argument(
        "--sub_domain_anme",
        type=str,
        default="1000_attribute_retranslated_with_mask",
    )
    parser.add_argument(
        "--vis_path",
        type=str,
        default="IDEA-Research/HumanRef/visualize",
    )
    parser.add_argument("--num_images", type=int, default=100)
    parser.add_argument("--vis_mask", type=bool, default=True)
    return parser.parse_args()


class ColorGenerator:

    def __init__(self, color_type) -> None:
        self.color_type = color_type

        if color_type == "same":
            self.color = tuple((np.random.randint(0, 127, size=3) + 128).tolist())
        elif color_type == "text":
            np.random.seed(3396)
            self.num_colors = 300
            self.colors = np.random.randint(0, 127, size=(self.num_colors, 3)) + 128
        else:
            raise ValueError

    def get_color(self, text):
        if self.color_type == "same":
            return self.color

        if self.color_type == "text":
            text_hash = hash(text)
            index = text_hash % self.num_colors
            color = tuple(self.colors[index])
            return color

        raise ValueError


def encode_counts_if_needed(rle):
    if isinstance(rle["counts"], list):
        return coco_mask.frPyObjects(rle, rle["size"][0], rle["size"][1])
    return rle


def convert_coco_rle_to_mask(segmentations, height, width):
    def process_polygon(polygon):
        polygon = encode_counts_if_needed(polygon)
        mask = coco_mask.decode(polygon)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        return mask

    with concurrent.futures.ThreadPoolExecutor() as executor:
        masks = list(executor.map(process_polygon, segmentations))

    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)

    return masks


if __name__ == "__main__":
    args = get_args()
    anno_path = args.anno_path
    with open(anno_path, "r") as f:
        annos = [json.loads(line) for line in f]
    annos = [
        anno
        for anno in annos
        if anno["domain"] == args.domain_anme
        and anno["sub_domain"] == args.sub_domain_anme
    ]
    # shuffle the dataset
    random.shuffle(annos)
    vis_num = args.num_images
    args.vis_path = f"{args.vis_path}/{args.domain_anme}_{args.sub_domain_anme}"
    if not os.path.exists(args.vis_path):
        os.makedirs(args.vis_path)
    # generate a random list of images
    font_path = "tools/Tahoma.ttf"
    font_size = 32
    boxwidth = 8
    font = ImageFont.truetype(font_path, font_size)
    color_generaor = ColorGenerator("text")
    raw_annos = []
    for i in tqdm(range(vis_num)):
        anno = annos[i]
        image_name = anno["image_name"]
        image_path = os.path.join(args.image_root_dir, image_name)
        candidate_boxes = anno["candidate_boxes"]
        answer_boxes = anno["answer_boxes"]
        answer_segmentations = anno["answer_segmentations"]
        referring = anno["referring"]
        max_words_per_line = 6
        words = referring.split()
        lines = []
        while len(words) > 0:
            line = " ".join(words[:max_words_per_line])
            lines.append(line)
            words = words[max_words_per_line:]
        referring = "\n".join(lines)
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        # first draw the candidate boxes
        first_person = True
        for box in answer_boxes:
            x0, y0, x1, y1 = box
            draw.rectangle(
                [x0, y0, x1, y1],
                outline=color_generaor.get_color(referring),
                width=boxwidth,
            )
            bbox = draw.textbbox((x0, y0), referring, font)
            box_h = bbox[3] - bbox[1]
            box_w = bbox[2] - bbox[0]

            y0_text = y0 - box_h - (boxwidth * 2)
            y1_text = y0 + boxwidth
            if y0_text < 0:
                y0_text = 0
                y1_text = y0 + 2 * boxwidth + box_h
            if first_person:
                draw.rectangle(
                    [x0, y0_text, bbox[2] + boxwidth * 2, y1_text],
                    fill=color_generaor.get_color(referring),
                )
                draw.text(
                    (x0 + boxwidth, y0_text),
                    str(referring),
                    fill="black",
                    font=font,
                )
                first_person = False

        # now draw the mask
        if args.vis_mask:
            h, w = image.size
            masks = convert_coco_rle_to_mask(answer_segmentations, h, w)
            rgba_image = image.convert("RGBA")
            for mask in masks:
                import random

                mask_color = (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                )

                # Convert the tensor mask to a PIL image
                mask_pil = Image.fromarray(
                    (mask.numpy() * 255).astype(np.uint8)
                ).convert("L")
                colored_mask = Image.new("RGBA", image.size)
                draw = ImageDraw.Draw(colored_mask)
                draw.bitmap(
                    (0, 0), mask_pil, fill=mask_color + (127,)
                )  # Adding semi-transparency

                # Composite the colored mask with the original image
                rgba_image = Image.alpha_composite(rgba_image, colored_mask)
            image = rgba_image.convert("RGB")
        image_name = anno["image_name"]
        image.save(os.path.join(args.vis_path, image_name))
