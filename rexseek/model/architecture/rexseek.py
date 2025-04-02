from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen2Model,
)
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

from rexseek.model.projector.builder import build_vision_projector
from rexseek.model.vision_encoder.builder import build_vision_tower
from rexseek.model.vision_encoder.convnext import ConvNextVisionEncoder
from rexseek.model.visual_prompt_encoder import MultiLevelROIVisualPrompt
from rexseek.utils.constants import (
    DEFAULT_GROUNDING_END,
    DEFAULT_GROUNDING_OBJECTS_END,
    DEFAULT_GROUNDING_OBJECTS_START,
    DEFAULT_GROUNDING_START,
    DEFAULT_OBJECT_INDEX,
    DEFAULT_OBJECT_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    MAX_NUM_OBJECTS,
)


def get_token_slices(input_ids):
    """
    Get slices of tokens based on special markers in the input tensor.

    Args:
        input_ids (torch.Tensor): A tensor of token IDs where IMAGE_TOKEN_INDEX represents an image token,
            DEFAULT_OBJECT_INDEX represents an object token, and all other values represent text tokens.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries where each dictionary contains the type of the
            token slice ('text', 'image', 'object') and the span as a list of start and end indices.
    """
    # define type markers and corresponding types
    type_map = {IMAGE_TOKEN_INDEX: "image", DEFAULT_OBJECT_INDEX: "object"}

    # find the positions of special markers
    image_indices = torch.where(input_ids == IMAGE_TOKEN_INDEX)[0]
    object_indices = torch.where(input_ids == DEFAULT_OBJECT_INDEX)[0]
    if len(object_indices) > 0:
        has_object = True
    else:
        has_object = False

    # merge all the positions of special markers
    special_indices = torch.cat((image_indices, object_indices))
    special_indices, _ = torch.sort(special_indices)
    special_tokens = input_ids[special_indices]

    slices = []
    start_idx = 0

    for i, idx in enumerate(special_indices):
        if start_idx < idx:
            slices.append({"type": "text", "span": [start_idx, idx.item()]})
        token_type = type_map[special_tokens[i].item()]
        slices.append({"type": token_type, "span": [idx.item(), idx.item() + 1]})
        start_idx = idx.item() + 1

    #
    if start_idx < len(input_ids):
        slices.append({"type": "text", "span": [start_idx, len(input_ids)]})

    return slices, has_object


class RexSeekQwenConfig(Qwen2Config):
    model_type = "rexseek_qwen"


class RexSeekQwenForCausalLM(Qwen2ForCausalLM):
    config_class = RexSeekQwenConfig

    def __init__(self, config, delay_load=True):
        # this config is passed from the pre-trained ckpt. If it's a config from the LLM, it won't
        # have key of mm_vision_tower, so we will not initialize the vision part, instead it will
        # be initialized later by calling self.initialize_vision_modules()
        super(RexSeekQwenForCausalLM, self).__init__(config)
        self.model = Qwen2Model(config)  # llm
        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(
                config, freeze_vision_tower=False, delay_load=delay_load
            )  # vision_tower
            self.vision_tower_aux = ConvNextVisionEncoder()
            self.mm_projector = build_vision_projector(
                config, start_hidden_size=2560
            )  # projector for vision_tower
            # projector for object token
            self.mm_object_projector = build_vision_projector(
                config, start_hidden_size=2880
            )

        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.box_encoder = MultiLevelROIVisualPrompt(
            output_size=7,
            channel_per_level=[192, 384, 768, 1536],  # ConvNeXt Large
            spatail_scale=192 / 768,
            add_pos_embedding=True,
            pos_embedding_dim=2880,
        )
        self.post_init()
        print("model initialized")

    def get_model(self):
        return self.model

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_vision_tower_aux(self):
        vision_tower_aux = getattr(self, "vision_tower_aux", None)
        if type(vision_tower_aux) is list:
            vision_tower_aux = vision_tower_aux[0]
        return vision_tower_aux

    def initialize_vision_modules(self, model_args, freeze_vision_tower, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        # load low resolution vision tower
        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(
                model_args, freeze_vision_tower=freeze_vision_tower
            )

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            if not vision_tower.is_loaded:
                vision_tower.load_model()

        # load high resolution vision tower
        if self.get_vision_tower_aux() is None:
            vision_tower_aux = build_vision_tower_aux(
                model_args, freeze_vision_tower=freeze_vision_tower
            )

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower_aux = [vision_tower_aux]
            else:
                self.vision_tower_aux = vision_tower_aux
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower_aux = self.vision_tower_aux[0]
            else:
                vision_tower_aux = self.vision_tower_aux
            if not vision_tower.is_loaded:
                vision_tower_aux.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(
            model_args, "mm_projector_type", "linear"
        )
        self.config.mm_hidden_size = 2560  # ! HARD CODE HERER
        self.config.object_hidden_size = 2880  # ! HARD CODE HERER
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type
        self.config.vis_during_training_prob = model_args.vis_during_training_prob

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(
                self.config, self.config.mm_hidden_size
            )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if getattr(self, "mm_object_projector", None) is None:
            self.mm_object_projector = build_vision_projector(
                self.config, self.config.object_hidden_size
            )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_object_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(
                pretrain_mm_mlp_adapter, map_location="cpu"
            )

            def get_w(weights, keyword):
                return {
                    k.split(keyword + ".")[1]: v
                    for k, v in weights.items()
                    if keyword in k
                }

            self.mm_projector.load_state_dict(
                get_w(mm_projector_weights, "mm_projector")
            )

            self.mm_object_projector.load_state_dict(
                get_w(mm_projector_weights, "mm_object_projector")
            )

    def encode_images(self, images, images_aux):
        low_res_feat = self.get_vision_tower()(images)
        aux_output = self.get_vision_tower_aux()(images_aux)
        visual_outputs_aux = aux_output["image_features"]
        high_res_feat = aux_output["last_feat"]  # (B, 1536, 24, 24)
        # concat the low res features with the high res features
        b, c, h, w = high_res_feat.shape  # (2, 1536, 24, 24)
        _, _, d = low_res_feat.shape  # (2, 576, 1024)
        high_res_feat = high_res_feat.view(b, c, h * w).transpose(1, 2)
        image_features = torch.cat((low_res_feat, high_res_feat), dim=-1)
        image_features = self.mm_projector(image_features)
        return image_features, visual_outputs_aux

    def encode_objects(
        self, bboxes, visual_outputs_aux, dtype, num_gt_boxes_per_image=None
    ):
        """Encode object features from bounding boxes.

        Args:
            bboxes (torch.Tensor): bounding boxes in the shape of (N, 4)
            image_features_before_proj (torch.Tensor): image features in the shape of (N, hidden_size)

        Returns:
            torch.Tensor: object features in the shape of (N, hidden_size)
        """
        bbox_visual_outputs = []
        for batch_idx, boxes in enumerate(bboxes):
            num_box = (
                num_gt_boxes_per_image[batch_idx]
                if num_gt_boxes_per_image is not None
                else len(boxes)
            )
            boxes = boxes[:num_box]
            if len(boxes) == 0:
                bbox_visual_outputs.append(None)
                continue
            multi_level_aux_features = [
                visual_output_aux[batch_idx].unsqueeze(0)
                for visual_output_aux in visual_outputs_aux
            ]
            out_vp_feat = self.box_encoder(
                multi_level_aux_features,
                [boxes],
            ).squeeze(0)
            out_vp_feat = out_vp_feat.to(dtype)
            out_vp_feat = self.mm_object_projector(out_vp_feat)
            bbox_visual_outputs.append(out_vp_feat)
        # b,n,c
        return bbox_visual_outputs

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        pixel_values=None,
        pixel_values_aux=None,
        gt_boxes=None,
        num_gt_boxes_per_image=None,
    ):
        if pixel_values is None:
            return (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                None,
                labels,
            )
        pixel_values, visual_outputs_aux = self.encode_images(
            pixel_values, pixel_values_aux
        )  # (B, 576, 2048)
        if gt_boxes is not None:
            bbox_feats = self.encode_objects(
                gt_boxes, visual_outputs_aux, pixel_values.dtype, num_gt_boxes_per_image
            )
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()  # padding mask in shaoe (B, L)
        if position_ids is None:
            position_ids = torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            )
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask)
        ]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        cur_object_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = pixel_values[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat(
                    [cur_input_embeds_1, cur_image_features[0:0]], dim=0
                )
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                cur_object_idx += 1
                continue

            cur_labels = labels[batch_idx]
            token_slices, has_object = get_token_slices(cur_input_ids)
            result_input_embeddings = []
            result_output_labels = []
            cur_gt_bnox_indice = 0
            cur_object_features = None
            for slice in token_slices:
                slice_type = slice["type"]
                slice_span = slice["span"]
                if slice_type == "text":
                    cur_input_ids_noim = cur_input_ids[slice_span[0] : slice_span[1]]
                    cur_labels_noim = cur_labels[slice_span[0] : slice_span[1]]
                    cur_input_embeds = self.get_model().embed_tokens(cur_input_ids_noim)
                    result_input_embeddings.append(cur_input_embeds)
                    result_output_labels.append(cur_labels_noim)
                elif slice_type == "image":
                    cur_input_embeds = pixel_values[cur_image_idx]
                    result_input_embeddings.append(cur_input_embeds)
                    result_output_labels.append(
                        torch.full(
                            (cur_input_embeds.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )
                    cur_image_idx += 1
                elif slice_type == "object":
                    try:
                        result_input_embeddings.append(
                            bbox_feats[cur_object_idx][cur_gt_bnox_indice].unsqueeze(0)
                        )
                    except:
                        raise ValueError(
                            f"current boxe_feats.shape: {bbox_feats[cur_object_idx].shape}, "
                        )
                    cur_gt_bnox_indice += 1
                    result_output_labels.append(
                        torch.full(
                            (1,),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )
            cur_object_idx += 1
            result_input_embeddings = torch.cat(result_input_embeddings)
            result_output_labels = torch.cat(result_output_labels)
            assert len(result_output_labels) == len(result_input_embeddings)
            new_input_embeds.append(result_input_embeddings)
            new_labels.append(result_output_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(
            self.config, "tokenizer_model_max_length", None
        )
        if tokenizer_model_max_length is not None:
            new_input_embeds = [
                x[:tokenizer_model_max_length] for x in new_input_embeds
            ]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        position_ids = torch.zeros(
            (batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device
        )

        for i, (cur_new_embed, cur_new_labels) in enumerate(
            zip(new_input_embeds, new_labels)
        ):
            cur_len = cur_new_embed.shape[0]
            new_input_embeds_padded.append(
                torch.cat(
                    (
                        cur_new_embed,
                        torch.zeros(
                            (max_len - cur_len, cur_new_embed.shape[1]),
                            dtype=cur_new_embed.dtype,
                            device=cur_new_embed.device,
                        ),
                    ),
                    dim=0,
                )
            )
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_new_labels
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(
                    0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return (
            None,
            position_ids,
            attention_mask,
            past_key_values,
            new_input_embeds,
            new_labels,
        )

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        object_tokens = [
            DEFAULT_OBJECT_TOKEN.replace("<i>", str(i)) for i in range(MAX_NUM_OBJECTS)
        ]
        add_tokens = object_tokens + [
            DEFAULT_GROUNDING_START,
            DEFAULT_GROUNDING_END,
            DEFAULT_GROUNDING_OBJECTS_START,
            DEFAULT_GROUNDING_OBJECTS_END,
        ]
        num_new_tokens = tokenizer.add_tokens(
            add_tokens,
            special_tokens=True,
        )
        if num_new_tokens > 0:
            self.resize_token_embeddings(len(tokenizer))
            # using the average of all the existed tokens to initialize the new tokens
            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True
            )
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True
            )

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg
            for p in self.get_input_embeddings().parameters():
                p.requires_grad = True
            for p in self.get_output_embeddings().parameters():
                p.requires_grad = True

        if model_args.pretrain_mm_mlp_adapter:
            mm_projector_weights = torch.load(
                model_args.pretrain_mm_mlp_adapter, map_location="cpu"
            )
            embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
            assert num_new_tokens == 2
            if input_embeddings.shape == embed_tokens_weight.shape:
                input_embeddings[-num_new_tokens:] = embed_tokens_weight[
                    -num_new_tokens:
                ]
            elif embed_tokens_weight.shape[0] == num_new_tokens:
                input_embeddings[-num_new_tokens:] = embed_tokens_weight
            else:
                raise ValueError(
                    f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_aux: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        gt_boxes: Optional[torch.LongTensor] = None,
        num_gt_boxes_per_image: Optional[torch.LongTensor] = None,
        tokenizer=None,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                pixel_values,
                pixel_values_aux,
                gt_boxes,
                num_gt_boxes_per_image,
            )

        # visualize the prediction during training
        if torch.rand(1).item() < self.config.vis_during_training_prob:
            with torch.inference_mode():
                output_ids = self.generate(
                    input_ids,
                    pixel_values,
                    pixel_values_aux,
                    image_sizes,
                    position_ids,
                    attention_mask,
                    gt_boxes=gt_boxes,
                    num_gt_boxes_per_image=num_gt_boxes_per_image,
                )
                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=False)
                print(outputs)
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor],
        pixel_values: Optional[torch.Tensor],
        pixel_values_aux: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:

        if inputs_embeds is None:
            position_ids = kwargs.pop("position_ids", None)
            attention_mask = kwargs.pop("attention_mask", None)
            gt_boxes = kwargs.pop("gt_boxes", None)
            num_gt_boxes_per_image = kwargs.pop("num_gt_boxes_per_image", None)

            if pixel_values is not None:
                (inputs, position_ids, attention_mask, _, inputs_embeds, _) = (
                    self.prepare_inputs_labels_for_multimodal(
                        inputs,
                        position_ids,
                        attention_mask,
                        past_key_values=None,
                        labels=None,
                        pixel_values=pixel_values,
                        pixel_values_aux=pixel_values_aux,
                        gt_boxes=gt_boxes,
                        num_gt_boxes_per_image=num_gt_boxes_per_image,
                    )
                )

            else:
                inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("rexseek_qwen", RexSeekQwenConfig)
AutoModelForCausalLM.register(RexSeekQwenConfig, RexSeekQwenForCausalLM)
