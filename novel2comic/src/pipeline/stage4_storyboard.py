# -*- coding: utf-8 -*-
"""Stage ④: 分镜生成——为每个场景设计格子化的画面语言。"""

from src.models import ChapterData, Scene, Panel
from src.llm_adapter import LLMAdapter
from src.img_adapter import ImageGenAdapter


SYSTEM_PROMPT = """你是专业的漫画分镜师 (Comic Storyboard Artist)，精通日本漫画和韩式条漫的分镜设计。

你需要为给定的场景生成完整的分镜脚本。每个场景生成 3-6 个格子。

每个格子 (Panel) 必须包含以下字段：
{
  "panel_number": 1,
  "visual_description": "中文画面描述，必须有构图信息（前景是什么、中景是什么、背景是什么）",
  "character_action": "该格中角色的动作和表情",
  "dialogue": "该格的台词内容（无台词则为空字符串）",
  "camera_angle": "镜头角度（特写/近景/中景/远景/俯视/仰视/主观POV/鸟瞰）",
  "mood": "该格的情绪氛围",
  "sd_prompt": "英文 Stable Diffusion prompt，必须包含: 画风关键词 + 画幅比例 + 角色描述 + 场景描述 + 构图提示"
}

分镜质量要求：
1. 每场景 3-6 格，动作场景可到 8 格
2. 画面描述必须有构图感（前/中/背景）
3. sd_prompt 必须包含画风关键词
4. 关键对话不能遗漏
5. 人物首次出场时描述外貌，后续用名字
6. 相邻格之间要有视觉变化（景别切换/视角变化），避免单调

以 JSON 数组格式返回所有格子的分镜数据。"""


def _build_sd_prompt(panel: dict, style_profile, characters: list) -> str:
    """自动增强 sd_prompt：注入风格基座 + 角色触发词。"""
    parts = []

    # 1. 风格基座
    if style_profile:
        parts.append(style_profile.sd_base_prompt)

    # 2. 角色触发词（自动注入出现角色的 trigger_words）
    if panel.get("character_refs"):
        for ref_name in panel["character_refs"]:
            for char in characters:
                if char.name == ref_name and char.sd_trigger_words:
                    parts.append(char.sd_trigger_words)

    # 3. 用户 prompt（LLM 生成的）
    if panel.get("sd_prompt"):
        parts.append(panel["sd_prompt"])

    # 4. 画幅比例
    if style_profile:
        parts.append(f"aspect ratio {style_profile.aspect_ratio}")

    return ", ".join(parts)


def run_stage4(data: ChapterData, llm: LLMAdapter, img_gen: ImageGenAdapter) -> ChapterData:
    """④ 分镜生成——逐场景生成。"""

    if not data.scenes:
        raise ValueError("Stage 3 (scenes) must run before Stage 4")

    char_info = "\n".join(
        f"- {c.name} [{c.role}]: {c.appearance.distinctive_features} | trigger: {c.sd_trigger_words}"
        for c in data.characters
    )

    for scene in data.scenes:
        print(f"\n  [SCENE] 处理场景 {scene.id}: {scene.title}")

        # 找到场景原文对应的段落
        scene_chars = scene.characters_in_scene
        relevant_lines = []
        for line in data.source_text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if any(ch in line for ch in scene_chars):
                relevant_lines.append(line)
        scene_text = "\n".join(relevant_lines[:20])

        user_prompt = (
            f"## 场景信息\n"
            f"- 标题: {scene.title}\n"
            f"- 摘要: {scene.summary}\n"
            f"- 情绪: {scene.emotion_arc}\n"
            f"- 关键台词: {scene.key_dialogue}\n\n"
            f"## 场景原文\n{scene_text}\n\n"
            f"## 角色信息\n{char_info}\n\n"
            f"## 风格\n{data.style_profile.name if data.style_profile else 'auto'}\n\n"
            f"请为此场景生成 3-6 格分镜脚本（JSON 数组）。"
        )

        result = llm.chat_json(SYSTEM_PROMPT, user_prompt)

        scene.panels = []
        for panel_dict in result:
            # 自动注入风格基座和角色触发词
            enhanced_prompt = _build_sd_prompt(
                panel_dict, data.style_profile, data.characters
            )

            panel = Panel(
                panel_number=panel_dict.get("panel_number", len(scene.panels) + 1),
                visual_description=panel_dict.get("visual_description", ""),
                character_action=panel_dict.get("character_action", ""),
                dialogue=panel_dict.get("dialogue", ""),
                camera_angle=panel_dict.get("camera_angle", ""),
                mood=panel_dict.get("mood", ""),
                sd_prompt=enhanced_prompt,
                character_refs=panel_dict.get("character_refs", scene.characters_in_scene),
            )
            scene.panels.append(panel)

        print(f"    生成 {len(scene.panels)} 格")
        for p in scene.panels:
            print(f"      格{p.panel_number}: [{p.camera_angle}] {p.visual_description[:40]}...")

    total_panels = sum(len(s.panels) for s in data.scenes)
    print(f"\n  总计 {total_panels} 格分镜")
    return data
