import argparse

import gradio as gr
import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

from demos.rexseek_grounding_dino_spacy_sam import (
    convert_all_cate_prediction_to_ans, inference_gdino, inference_rexseek,
    inference_sam, load_model, load_rexseek_model, modify_processor_resolution,
    spacy_noun_phrases, visualize_rexseek_output)
from rexseek.tools.visualize import visualize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rexseek-path", type=str, default="IDEA-Research/RexSeek-3B")
    parser.add_argument(
        "--gdino-config",
        type=str,
        default="demos/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    )
    parser.add_argument(
        "--gdino-weights",
        type=str,
        default="demos/GroundingDINO/weights/groundingdino_swint_ogc.pth",
    )
    parser.add_argument(
        "--sam-weights",
        type=str,
        default="demos/segment-anything/weights/sam_vit_h_4b8939.pth",
    )
    parser.add_argument(
        "--server-ip",
        type=str,
        default="192.168.81.133",
        help="IP address to bind the server to",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=2512,
        help="Port to run the server on",
    )
    return parser.parse_args()


def initialize_models(args):
    # Load GDINO model
    gdino_model = load_model(args.gdino_config, args.gdino_weights).to("cuda")

    # Load SAM model
    sam = sam_model_registry["vit_h"](checkpoint=args.sam_weights)
    sam.to("cuda")
    sam_predictor = SamPredictor(sam)

    # Load RexSeek model
    tokenizer, rexseek_model, image_processor, _ = load_rexseek_model(args.rexseek_path)
    image_processor, crop_size_raw = modify_processor_resolution(image_processor)

    return (
        gdino_model,
        sam_predictor,
        tokenizer,
        rexseek_model,
        image_processor,
        crop_size_raw,
    )


def process_image(
    image,
    gdino_prompt,
    rexseek_prompt,
    text_threshold,
    box_threshold,
    gdino_model,
    sam_predictor,
    tokenizer,
    rexseek_model,
    image_processor,
    crop_size_raw,
):
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Run GDINO inference
    candidate_boxes = inference_gdino(
        image,
        [gdino_prompt],
        gdino_model,
        TEXT_TRESHOLD=float(text_threshold),
        BOX_TRESHOLD=float(box_threshold),
    )

    # Visualize GDINO results
    print(candidate_boxes)
    gdino_vis = visualize(
        image, candidate_boxes, [gdino_prompt for _ in range(len(candidate_boxes))]
    )

    # Run RexSeek inference
    answer = inference_rexseek(
        image,
        image_processor,
        tokenizer,
        rexseek_model,
        crop_size_raw,
        candidate_boxes,
        f"Please detect {rexseek_prompt} in this image. Answer the question with object indexes.",
    )

    # Process RexSeek output
    ans = convert_all_cate_prediction_to_ans(answer)
    pred_boxes = []
    for k, v in ans.items():
        for box in v:
            obj_idx = int(box[4:-1])
            if obj_idx < len(candidate_boxes):
                pred_boxes.append(candidate_boxes[obj_idx])

    if len(pred_boxes) > 0:
        # Run SAM inference
        masks = inference_sam(image, sam_predictor, torch.cat(pred_boxes, 0))
        # Final visualization
        final_vis = visualize_rexseek_output(
            image,
            input_boxes=candidate_boxes,
            masks=masks,
            prediction_text=answer,
        )
    else:
        final_vis = image

    return gdino_vis, final_vis, answer


def create_demo(models):
    (
        gdino_model,
        sam_predictor,
        tokenizer,
        rexseek_model,
        image_processor,
        crop_size_raw,
    ) = models

    with gr.Blocks() as demo:
        gr.Markdown("# RexSeek + GroundingDINO + SAM Demo")

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input Image", type="pil")
                gdino_prompt = gr.Textbox(
                    label="GroundingDINO Prompt",
                    placeholder="person",
                    value="person",
                )
                rexseek_prompt = gr.Textbox(
                    label="RexSeek Prompt",
                    placeholder="person wearning red shirt and a black hat",
                    value="person wearning red shirt and a black hat",
                )
                text_threshold = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.25,
                    step=0.01,
                    label="Text Threshold for GroundingDINO",
                )
                box_threshold = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.25,
                    step=0.01,
                    label="Box Threshold for GroundingDINO",
                )
                run_button = gr.Button("Run")

            with gr.Column():
                gdino_output = gr.Image(label="GroundingDINO Detection")
                final_output = gr.Image(label="Final Result (RexSeek + SAM)")
                llm_output = gr.Textbox(label="LLM Raw Output", interactive=False)

        # Add examples section
        gr.Markdown("## Examples")
        examples = gr.Examples(
            examples=[
                [
                    "demos/demo_images/demo1.jpg",
                    "person",
                    "person that is giving a proposal",
                    0.25,
                    0.25,
                ],
                [
                    "demos/demo_images/demo2.jpg",
                    "person",
                    "prisoners",
                    0.25,
                    0.25,
                ],
                [
                    "demos/demo_images/demo3.jpg",
                    "person",
                    "Mr Alvin KWOCK",
                    0.25,
                    0.25,
                ],
                [
                    "demos/demo_images/demo4.jpg",
                    "person",
                    "the person next to Trump",
                    0.25,
                    0.25,
                ],
                [
                    "demos/demo_images/demo5.jpg",
                    "person",
                    "homelander",
                    0.25,
                    0.25,
                ],
                [
                    "demos/demo_images/demo6.jpg",
                    "person",
                    "male",
                    0.25,
                    0.25,
                ],
                [
                    "demos/demo_images/demo7.jpg",
                    "person",
                    "female",
                    0.25,
                    0.25,
                ],
                [
                    "demos/demo_images/demo8.jpg",
                    "person",
                    "walter white",
                    0.25,
                    0.25,
                ],
                [
                    "demos/demo_images/demo9.jpg",
                    "tomato",
                    "unripe tomato",
                    0.25,
                    0.25,
                ],
                [
                    "demos/demo_images/demo10.jpeg",
                    "pigeon",
                    "pigeon on the ground",
                    0.25,
                    0.25,
                ],
            ],
            inputs=[
                input_image,
                gdino_prompt,
                rexseek_prompt,
                text_threshold,
                box_threshold,
            ],
            outputs=[gdino_output, final_output, llm_output],
            fn=lambda img, p1, p2, tt, bt: process_image(
                img,
                p1,
                p2,
                tt,
                bt,
                gdino_model,
                sam_predictor,
                tokenizer,
                rexseek_model,
                image_processor,
                crop_size_raw,
            ),
            cache_examples=False,
        )

        run_button.click(
            fn=lambda img, p1, p2, tt, bt: process_image(
                img,
                p1,
                p2,
                tt,
                bt,
                gdino_model,
                sam_predictor,
                tokenizer,
                rexseek_model,
                image_processor,
                crop_size_raw,
            ),
            inputs=[
                input_image,
                gdino_prompt,
                rexseek_prompt,
                text_threshold,
                box_threshold,
            ],
            outputs=[gdino_output, final_output, llm_output],
        )

    return demo


def main():
    args = parse_args()
    models = initialize_models(args)
    demo = create_demo(models)
    demo.launch(server_name=args.server_ip, server_port=args.server_port, share=True)


if __name__ == "__main__":
    main()
