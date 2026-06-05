# -*- coding: utf-8 -*-
"""Stage ①: 文本分析——识别风格、人物预览、情感基调。"""

from src.models import ChapterData, AnalysisResult
from src.llm_adapter import LLMAdapter
from src.img_adapter import ImageGenAdapter
from src.styles import detect_style


SYSTEM_PROMPT = """你是一位资深的小说编辑和漫画改编顾问。
你需要分析一段小说文本，提取关键信息用于后续的漫画改编。

请分析以下维度并以 JSON 格式返回：
{
  "genre_tags": ["题材标签1", "题材标签2", ...],
  "style": "manga 或 webtoon 或 gufeng",
  "tone": ["情感基调1", "情感基调2", ...],
  "era": "时代背景",
  "pace": "叙事节奏（快节奏/慢热/张弛有度）",
  "characters_preview": [
    {"name": "角色名", "role": "主角/配角/反派/路人", "first_appearance_line": "首次出场的原文片段"}
  ]
}

题材标签从下列中选择：武侠, 仙侠, 玄幻, 都市, 校园, 科幻, 悬疑, 历史, 言情, 轻小说, 古装, 日常, 异世界, 职场, 恋爱

风格判断规则：
- 武侠/仙侠/玄幻/历史/古装 → gufeng
- 轻小说/校园/恋爱/日常/异世界 → manga
- 都市/职场/现实/娱乐圈 → webtoon
- 科幻/悬疑 → 快节奏用 manga，慢节奏用 webtoon"""


def run_stage1(data: ChapterData, llm: LLMAdapter, img_gen: ImageGenAdapter) -> ChapterData:
    """① 文本分析。"""

    # 长文本截断策略：取前 3000 字符分析
    text_sample = data.source_text[:3000]

    user_prompt = f"请分析以下小说片段：\n\n{text_sample}"

    result = llm.chat_json(SYSTEM_PROMPT, user_prompt)

    data.analysis = AnalysisResult(
        genre_tags=result.get("genre_tags", []),
        style=result.get("style", "auto"),
        tone=result.get("tone", []),
        era=result.get("era", ""),
        pace=result.get("pace", ""),
        characters_preview=result.get("characters_preview", []),
    )

    # 自动判断风格
    detected = detect_style(data.analysis.genre_tags, data.analysis.pace)
    data.analysis.style = detected.name

    # 设置 StyleProfile
    data.style_profile = detected

    print(f"  题材: {data.analysis.genre_tags}")
    print(f"  风格: {data.analysis.style}")
    print(f"  基调: {data.analysis.tone}")
    print(f"  人物预览: {[c['name'] for c in data.analysis.characters_preview]}")

    return data
