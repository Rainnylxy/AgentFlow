# -*- coding: utf-8 -*-
"""Stage ②: 角色设计——提取外貌描述 + 生成 SD 触发词。"""

from src.models import ChapterData, CharacterSheet, CharacterAppearance
from src.llm_adapter import LLMAdapter
from src.img_adapter import ImageGenAdapter


SYSTEM_PROMPT = """你是专业的漫画角色设计师。根据小说文本和分析结果，为每个角色创建详细的 Character Sheet。

对每个角色，提取或推断以下信息并以 JSON 数组格式返回：
[
  {
    "id": "英文名_小写_下划线",
    "name": "中文名",
    "role": "protagonist/antagonist/supporting/minor",
    "appearance": {
      "face": "脸型、五官特征、肤色",
      "hair": "发型、发色、长度",
      "build": "体型（高矮胖瘦）",
      "clothing": "服装风格和细节",
      "accessories": "配饰（武器、首饰等）",
      "distinctive_features": "最独特的视觉特征（一句话概括）"
    },
    "sd_trigger_words": "英文触发词，用于 Stable Diffusion 保持角色一致性。格式: 'name, gender, hair description, clothing, distinctive feature, art style neutral'",
    "personality_notes": "性格特征对表情/姿态的影响"
  }
]

重要：sd_trigger_words 必须足够详细，确保每次生图都能保持角色外貌一致。
首次出场的角色优先从原文中提取外貌描写，没有直接描写的合理推断。"""


def run_stage2(data: ChapterData, llm: LLMAdapter, img_gen: ImageGenAdapter) -> ChapterData:
    """② 角色设计。"""

    if not data.analysis:
        raise ValueError("Stage 1 (analysis) must run before Stage 2")

    # 已有角色库中的角色跳过重新设计
    existing_names = {c.name for c in data.characters}
    new_characters = [
        p for p in data.analysis.characters_preview
        if p["name"] not in existing_names
    ]

    if not new_characters:
        print("  所有角色已设计，跳过")
        return data

    # 提取外貌相关的原文段落（包含角色名的句子）
    relevant_lines = []
    for line in data.source_text.split("\n"):
        for char in new_characters:
            if char["name"] in line:
                relevant_lines.append(line.strip())
                break

    text_context = "\n".join(relevant_lines[:30])  # 最多 30 行

    user_prompt = (
        f"## 人物列表\n"
        + "\n".join(f"- {c['name']} ({c['role']})" for c in new_characters)
        + f"\n\n## 原文片段（含外貌描写）\n{text_context}\n\n"
        + f"## 风格\n{data.style_profile.name if data.style_profile else 'auto'}\n\n"
        + f"请为以上每个角色生成 Character Sheet（JSON 数组）。"
    )

    result = llm.chat_json(SYSTEM_PROMPT, user_prompt)

    for char_dict in result:
        appearance = CharacterAppearance(**char_dict.get("appearance", {}))
        sheet = CharacterSheet(
            id=char_dict.get("id", ""),
            name=char_dict.get("name", ""),
            role=char_dict.get("role", ""),
            appearance=appearance,
            sd_trigger_words=char_dict.get("sd_trigger_words", ""),
            personality_notes=char_dict.get("personality_notes", ""),
            status="draft",
        )
        data.characters.append(sheet)
        print(f"  👤 {sheet.name} [{sheet.role}] → trigger_words: {sheet.sd_trigger_words[:60]}...")

    return data
