# -*- coding: utf-8 -*-
"""Stage ③: 场景拆分——将小说文本切分为关键叙事场景。"""

from src.models import ChapterData, Scene
from src.llm_adapter import LLMAdapter
from src.img_adapter import ImageGenAdapter


SYSTEM_PROMPT = """你是专业的漫画改编编剧，擅长将小说文本拆分为适合漫画表现的场景。

一个"场景"是一个独立的叙事单元——地点不变、时间连续、有明确的起承转合。

拆分规则：
- 按地点变换切分（从街上到屋内 = 新场景）
- 按时间跳跃切分（"三天后" = 新场景）
- 按情绪转折切分（从平静到冲突爆发 = 可分新场景）
- 一章通常 3-8 个场景，不要超过 8 个

返回 JSON 数组：
[
  {
    "id": 1,
    "title": "场景标题",
    "summary": "1-2句话概述",
    "characters_in_scene": ["角色名1"],
    "emotion_arc": "情绪变化（如：平静→紧张）",
    "key_dialogue": "该场景最重要的台词"
  }
]"""


def run_stage3(data: ChapterData, llm: LLMAdapter, img_gen: ImageGenAdapter) -> ChapterData:
    """③ 场景拆分。"""

    char_names = [c.name for c in data.characters]
    style_name = data.style_profile.name if data.style_profile else "auto"

    user_prompt = (
        f"## 原文\n{data.source_text}\n\n"
        f"## 已识别角色\n{', '.join(char_names) if char_names else '（无）'}\n\n"
        f"## 风格\n{style_name}\n\n"
        f"请将以上文本拆分为关键场景（3-8个）。"
    )

    result = llm.chat_json(SYSTEM_PROMPT, user_prompt)

    data.scenes = []
    for scene_dict in result:
        scene = Scene(
            id=scene_dict.get("id", len(data.scenes) + 1),
            title=scene_dict.get("title", ""),
            summary=scene_dict.get("summary", ""),
            characters_in_scene=scene_dict.get("characters_in_scene", []),
            emotion_arc=scene_dict.get("emotion_arc", ""),
            key_dialogue=scene_dict.get("key_dialogue", ""),
        )
        data.scenes.append(scene)
        print(f"  🎬 场景{scene.id}: {scene.title} [{scene.emotion_arc}]")

    print(f"  共 {len(data.scenes)} 个场景")
    return data
