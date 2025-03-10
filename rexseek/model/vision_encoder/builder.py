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
