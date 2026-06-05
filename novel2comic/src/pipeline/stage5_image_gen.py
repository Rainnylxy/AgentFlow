# -*- coding: utf-8 -*-
"""Stage ⑤: 图像生成——为每格分镜生成对应的漫画图片。"""

import os
from src.models import ChapterData
from src.llm_adapter import LLMAdapter
from src.img_adapter import ImageGenAdapter


def _get_image_size(style_profile) -> tuple:
    """根据风格获取图片尺寸。"""
    ratio_map = {
        "9:16": (576, 1024),
        "4:3": (1024, 768),
        "16:9": (1024, 576),
        "1:1": (1024, 1024),
    }
    if style_profile:
        return ratio_map.get(style_profile.aspect_ratio, (1024, 1024))
    return (1024, 1024)


def run_stage5(data: ChapterData, llm: LLMAdapter, img_gen: ImageGenAdapter) -> ChapterData:
    """⑤ 图像生成——逐格调用生图 API 或生成占位图。"""

    width, height = _get_image_size(data.style_profile)
    images_dir = os.path.join(data.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    total_panels = sum(len(s.panels) for s in data.scenes)
    current = 0

    for scene in data.scenes:
        for panel in scene.panels:
            current += 1
            print(f"\n  🖼️  生成 [{current}/{total_panels}] 场景{scene.id} 格{panel.panel_number}")

            # 找该格涉及角色的参考图
            ref_path = ""
            for char_name in panel.character_refs:
                for char in data.characters:
                    if char.name == char_name and char.reference_image_path:
                        ref_path = char.reference_image_path
                        break

            image_path = img_gen.generate(
                prompt=panel.sd_prompt,
                output_dir=images_dir,
                width=width,
                height=height,
                reference_image_path=ref_path,
            )

            panel.generated_image_path = image_path
            panel.status = "generated"
            print(f"    → {os.path.basename(image_path)}")

    print(f"\n  全部 {total_panels} 格图片生成完毕")
    return data
