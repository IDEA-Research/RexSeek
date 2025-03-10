import os

from .clip_encoder import CLIPVisionTower
from .openclip_encoder import OpenCLIPVisionTower


def build_vision_tower(vision_tower_cfg, freeze_vision_tower=True, **kwargs):
    vision_tower = getattr(
        vision_tower_cfg,
        "mm_vision_tower",
        getattr(vision_tower_cfg, "vision_tower", None),
    )
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, "s2", False)
    if is_absolute_path_exists or vision_tower.startswith("openai"):
        return CLIPVisionTower(
            vision_tower,
            args=vision_tower_cfg,
            freeze_vision_tower=freeze_vision_tower,
            **kwargs,
        )


def build_vision_tower_aux(vision_tower_cfg, freeze_vision_tower=True, **kwargs):
    if not os.path.exists(
        "/comp_robot/jiangqing/LLaVA_Refactor/mounted_files/checkpoints/laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup"
    ):
        return OpenCLIPVisionTower(
            vision_tower="model_zoo/OpenAI/openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup",
            vision_tower_path="/comp_robot/jiangqing/projects/2023/research/LLaVA_Refactor/mounted_files/checkpoints/laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup",
            optimize_vision_tower_aux=not freeze_vision_tower,
            use_last_feat=True,
        )
    return OpenCLIPVisionTower(
        vision_tower="model_zoo/OpenAI/openclip-convnext-large-d-320-laion2B-s29B-b131K-ft-soup",
        vision_tower_path="/comp_robot/jiangqing/LLaVA_Refactor/mounted_files/checkpoints/laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup",
        optimize_vision_tower_aux=not freeze_vision_tower,
        use_last_feat=True,
    )
