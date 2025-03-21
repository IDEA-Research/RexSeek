import torch
from rexseek.model.architecture import RexSeekQwenForCausalLM
from rexseek.utils import (
    DEFAULT_GROUNDING_END,
    DEFAULT_GROUNDING_OBJECTS_END,
    DEFAULT_GROUNDING_OBJECTS_START,
    DEFAULT_GROUNDING_START,
    DEFAULT_OBJECT_TOKEN,
)
from transformers import AutoTokenizer, BitsAndBytesConfig


def load_rexseek_model(
    model_path,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    use_flash_attn=False,
    max_num_objects=100,
    **kwargs,
):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs["device_map"] = {"": device}

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    if use_flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    # add speicial tokens
    added_tokens = [
        DEFAULT_OBJECT_TOKEN.replace("<i>", str(i)) for i in range(max_num_objects)
    ]
    added_tokens += [
        DEFAULT_GROUNDING_START,
        DEFAULT_GROUNDING_END,
        DEFAULT_GROUNDING_OBJECTS_START,
        DEFAULT_GROUNDING_OBJECTS_END,
    ]
    # add tokens
    if len(added_tokens) > 0:
        for add_token in added_tokens:
            if add_token not in tokenizer.get_vocab():
                tokenizer.add_tokens([add_token], special_tokens=True)

    model = RexSeekQwenForCausalLM.from_pretrained(
        model_path, low_cpu_mem_usage=True, **kwargs
    )

    image_processor = None

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model(device_map=device_map)
    if device_map != "auto":
        vision_tower.to(device=device_map, dtype=torch.float16)

    # vision_tower_aux = model.get_vision_tower_aux()
    # if not vision_tower_aux.is_loaded:
    #     vision_tower_aux.load_model(device_map=device_map)
    # if device_map != "auto":
    #     vision_tower_aux.to(device=device_map, dtype=torch.float16)

    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
