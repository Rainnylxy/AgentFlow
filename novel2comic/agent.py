# -*- coding: utf-8 -*-
"""
Novel2Comic Agent
=================
基于 AgentFlow 框架，将小说文本转化为漫画分镜脚本 + 图片生成 prompt。

用法:
    python agent.py "小说文本内容"

环境变量:
    AGENTFLOW_API_KEY  - DeepSeek API key
    AGENTFLOW_BASE_URL - API 地址（默认 https://api.deepseek.com/v1）
    AGENTFLOW_MODEL    - 模型名（默认 deepseek-chat）
    AGENTFLOW_PROXY    - HTTP 代理（可选）
"""

import asyncio
import json
import os
import sys
from datetime import datetime

# 添加 AgentFlow 到路径（开发阶段）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentflow.runtime.builder import AgentBuilder
from agentflow.runtime.toolkit import tool
from agentflow.runtime.memory.manager import MemoryProfile
from agentflow.runtime.prompt import PromptTemplate
from agentflow.runtime.prompt.section import Section
from agentflow.runtime.thinking import ThinkingMode
from agentflow.runtime.llm_client import OpenAIClient


# ============================================================
# Prompt 模板：漫画分镜师
# ============================================================

class ComicStoryboardRole(Section):
    """角色定义：专业漫画分镜师"""
    name = "role_card"
    order = 10

    def render(self, context: dict) -> str:
        return (
            "## Role\n"
            "你是专业的漫画分镜师(Comic Storyboard Artist)，精通日本漫画(ネーム/Name)"
            "与韩式条漫(Webtoon)的分镜设计。"
            "你的任务是将小说文字转化为可视化分镜脚本，"
            "为每一格生成画面描述和 Stable Diffusion/DALL-E 生图 prompt。\n\n"
            "## 工作流程（严格按顺序）\n"
            "1. 调用 analyze_text 分析文本类型、风格、人物、基调\n"
            "2. 调用 extract_scenes 拆分为关键场景\n"
            "3. 对每个场景依次调用 storyboard_scene 生成分镜\n"
            "4. 调用 compile_chapter 汇总为完整 Markdown 分镜脚本\n"
            "注意：必须等待上一步工具返回结果后再执行下一步。"
        )


class ComicStyleGuide(Section):
    """漫画风格规范"""
    name = "style_guide"
    order = 20

    def render(self, context: dict) -> str:
        return (
            "## 漫画风格规范\n"
            "### 日式漫画 (manga)\n"
            "- 黑白为主，可点缀灰度网点\n"
            "- 注重视觉动线引导与特写/远景交替\n"
            "- 使用速度线、集中线、效果字增强表现力\n"
            "- 对话框形状随情绪变化\n\n"
            "### 彩色条漫 (webtoon)\n"
            "- 全彩色，柔和调色板\n"
            "- 竖屏滑动，每格宽度一致，人物居中偏上\n"
            "- 格间留白控制阅读节奏\n\n"
            "### 自动判断 (auto)\n"
            "- 轻小说/校园/恋爱 → manga\n"
            "- 网文/都市/职场 → webtoon\n"
            "- 武侠/玄幻/仙侠 → manga\n"
            "- 科幻/悬疑 → 根据节奏判断"
        )


class ComicOutputFormat(Section):
    """输出格式定义"""
    name = "output_format"
    order = 30

    def render(self, context: dict) -> str:
        return (
            "## 分镜输出格式\n"
            "每个 storyboard_scene 返回的 JSON 数组必须包含以下字段：\n"
            "- panel_number: 格号\n"
            "- visual_description: 画面中文描述（含前景/中景/背景）\n"
            "- character_action: 角色动作和表情\n"
            "- dialogue: 台词（无则留空）\n"
            "- camera_angle: 镜头角度的中文描述\n"
            "- mood: 情绪氛围\n"
            "- sd_prompt: 英文生图 prompt（含画风关键词 + 关键视觉元素）\n\n"
            "compile_chapter 最终输出为 Markdown：\n"
            "- # 章标题\n"
            "- ## 基础信息（风格、人物表）\n"
            "- ## 场景N（每场景二级标题）\n"
            "- ### 格N（每格三级标题）\n"
            "- 文末 ## SD Prompts 汇总"
        )


class ComicQualityRules(Section):
    """质量约束"""
    name = "quality_rules"
    order = 40

    def render(self, context: dict) -> str:
        return (
            "## 质量规范\n"
            "1. 每场景 3-6 格分镜\n"
            "2. 画面描述必须有构图信息（前景/中景/背景）\n"
            "3. sd_prompt 必须包含 anime style/manga style/webtoon style + 画幅比例 + 独有视觉元素\n"
            "4. 关键情感转折台词不能遗漏\n"
            "5. 人物首次出现描述外貌，后续用名字指代\n"
            "6. 相邻格有视觉变化（景别/视角切换）\n"
            "7. extract_scenes 的 max_scenes 上限为 8"
        )


def create_prompt_template() -> PromptTemplate:
    """组装漫画分镜 Prompt 模板。"""
    template = PromptTemplate("comic_storyboard")
    template.add(ComicStoryboardRole())
    template.add(ComicStyleGuide())
    template.add(ComicOutputFormat())
    template.add(ComicQualityRules())
    return template


# ============================================================
# 4 个 Tool
# ============================================================

@tool
def analyze_text(text: str) -> str:
    """分析小说文本的类型、风格、人物和情感基调。

    分析维度：
    - type: 小说类型（玄幻/都市/校园/科幻/悬疑/...）
    - style: 推荐漫画风格（manga 或 webtoon）
    - characters: 人物列表 [{name, role, traits}]
    - tone: 整体基调
    - era: 时代背景

    返回 JSON 格式分析结果。
    """
    return json.dumps({
        "status": "analyzed",
        "instruction": (
            "Analyze the provided text and return a JSON object with: "
            "type (genre), recommended_style (manga or webtoon), "
            "characters (list of {name, role, traits}), "
            "tone (emotional tone), era (time period). "
            "Output ONLY the JSON, no other text."
        )
    }, ensure_ascii=False)


@tool
def extract_scenes(text: str, max_scenes: int = 6) -> str:
    """将文本拆分为关键叙事场景。

    每个场景是一个有起承转合的完整叙事单元。

    Args:
        text: 小说文本
        max_scenes: 最多拆几个场景（默认 6，上限 8）

    返回 JSON 数组: [{id, title, summary, characters_involved, emotion, key_dialogue}]
    """
    return json.dumps({
        "status": "scenes_extracted",
        "instruction": (
            f"Extract up to {max_scenes} key narrative scenes from the text. "
            "For each scene output a JSON array of objects with fields: "
            "id (number), title (short descriptive name), "
            "summary (1-2 sentences of what happens), "
            "characters_involved (list of character names), "
            "emotion (dominant emotion), "
            "key_dialogue (the most important line, or empty string). "
            "Output ONLY the JSON array, no other text."
        )
    }, ensure_ascii=False)


@tool
def storyboard_scene(
    scene_summary: str,
    characters: str,
    style: str = "auto",
    panels_per_scene: int = 4
) -> str:
    """为一个场景生成完整的漫画分镜。

    根据场景摘要和人物信息，生成该场景每格的画面描述和生图 prompt。

    Args:
        scene_summary: 该场景的摘要
        characters: JSON 格式人物列表
        style: manga/webtoon/auto
        panels_per_scene: 分镜格数（3-6 之间）

    返回 JSON 数组:
    [{panel_number, visual_description, character_action, dialogue, camera_angle, mood, sd_prompt}]
    """
    return json.dumps({
        "status": "storyboard_ready",
        "instruction": (
            f"Generate a storyboard with {panels_per_scene} panels for this scene "
            f"in {style} style. "
            f"Scene: {scene_summary}. "
            f"Characters: {characters}. "
            "For each panel, output a JSON object with fields: "
            "panel_number (int), "
            "visual_description (Chinese text describing the composition — foreground, midground, background), "
            "character_action (what the character is doing, in Chinese), "
            "dialogue (the spoken line, or empty string), "
            "camera_angle (Chinese description of shot type — 特写/中景/远景/俯视/仰视etc), "
            "mood (emotional tone of this panel, in Chinese), "
            "sd_prompt (English Stable Diffusion prompt with anime/manga/webtoon style keywords, aspect ratio, key visual elements). "
            "Output ONLY the JSON array, no other text."
        )
    }, ensure_ascii=False)


@tool
def compile_chapter(
    chapter_title: str,
    scenes_storyboard: str,
    style: str = "auto"
) -> str:
    """将全部分镜汇总为最终的 Markdown 分镜脚本。

    Args:
        chapter_title: 章节标题
        scenes_storyboard: JSON 格式的全部分镜数据
        style: 漫画风格

    返回完整的 Markdown 格式分镜脚本文件内容。
    """
    return json.dumps({
        "status": "compiled",
        "instruction": (
            f"Compile all storyboards into a final Markdown document. "
            f"Title: '{chapter_title}', Style: {style}. "
            f"Use the following structure:\n"
            f"# {chapter_title}\n"
            f"## 基础信息\n(style, generated date placeholder, style rationale)\n"
            f"## 人物一览\n(table with columns: 姓名 | 身份 | 外貌/特征)\n"
            f"## 场景1: [场景标题]\n### 格1: [画面概述]\n"
            f"(then for each panel: 画面描述, 台词, 镜头, 情绪, **SD Prompt**: ...)\n"
            f"...repeat for all panels and all scenes...\n"
            f"## SD Prompts 汇总\n(numbered list of all prompts)\n\n"
            f"Scenes data: {scenes_storyboard}"
        )
    }, ensure_ascii=False)


# ============================================================
# Agent 构建
# ============================================================

def build_agent():
    """构建 Novel2Comic Agent。"""
    api_key = os.getenv("AGENTFLOW_API_KEY", "")
    base_url = os.getenv("AGENTFLOW_BASE_URL", "https://api.deepseek.com/v1")
    model = os.getenv("AGENTFLOW_MODEL", "deepseek-chat")
    proxy = os.getenv("AGENTFLOW_PROXY", "")

    if not api_key:
        print("[!] AGENTFLOW_API_KEY not set.")
        print("    请设置环境变量: export AGENTFLOW_API_KEY='sk-your-key'")
        print("    或复制 .env.example 为 .env 并填入你的 key")
        sys.exit(1)

    llm = OpenAIClient(
        api_key=api_key,
        model=model,
        base_url=base_url,
        proxy=proxy or None,
    )
    prompt = create_prompt_template()

    agent = (AgentBuilder("novel2comic")
        .with_llm(llm)
        .with_tools(analyze_text, extract_scenes, storyboard_scene, compile_chapter)
        .with_prompt(prompt)
        .with_memory(MemoryProfile.standard())
        .with_thinking(ThinkingMode.ADAPTIVE)
        .with_max_iterations(25)
        .build())

    return agent


async def run_novel2comic(text: str, title: str = "未命名章节") -> str:
    """执行小说→漫画分镜转换。

    Args:
        text: 小说文本
        title: 章节标题

    Returns:
        Markdown 格式的分镜脚本
    """
    agent = build_agent()

    task = (
        f"## 章节标题\n{title}\n\n"
        f"## 小说原文\n{text}\n\n"
        f"## 任务\n"
        f"请严格按以下顺序执行：\n"
        f"1. 调用 analyze_text 分析文本的类型、风格、人物、基调\n"
        f"2. 根据 analyze_text 返回的 recommended_style 确定风格，"
        f"调用 extract_scenes 拆分关键场景\n"
        f"3. 对每个场景依次调用 storyboard_scene 生成分镜"
        f"（使用上一步返回的 scenes 数组和 characters 列表）\n"
        f"4. 所有场景处理完毕后，调用 compile_chapter 汇总为完整的 Markdown 分镜脚本\n\n"
        f"风格默认为 auto（自动判断）。"
    )

    print(f"[novel2comic] 开始处理: {title}")
    print(f"[novel2comic] 文本长度: {len(text)} 字符")

    result = await agent.run(task)

    # 保存到 outputs 目录
    os.makedirs("outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = "".join(c if c.isalnum() or c in "._- " else "_" for c in title)
    output_file = f"outputs/{timestamp}_{safe_title}.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# Novel2Comic 分镜脚本\n\n")
        f.write(f"- **章节**: {title}\n")
        f.write(f"- **生成时间**: {datetime.now().isoformat()}\n")
        f.write(f"- **原文长度**: {len(text)} 字符\n\n")
        f.write("---\n\n")
        f.write(result.output)

    print(f"[novel2comic] 输出已保存: {output_file}")
    print(f"[novel2comic] Agent 执行步骤数: {len(result.steps)}")
    for i, step in enumerate(result.steps):
        step_type = step.get("type", step.get("phase", "?"))
        print(f"    步骤{i}: {step_type}")

    return result.output


# ============================================================
# CLI 入口
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Novel2Comic Agent - 小说转漫画分镜")
        print()
        print("用法: python agent.py <小说文本或文件路径> [章节标题]")
        print()
        print("示例:")
        print('  python agent.py "一个少年在月光下拔出了剑..." 第一章')
        print("  python agent.py chapter1.txt 月下初遇")
        sys.exit(1)

    input_text = sys.argv[1]
    chapter_title = sys.argv[2] if len(sys.argv) > 2 else "未命名章节"

    # 如果是文件路径，读取文件
    if os.path.isfile(input_text):
        with open(input_text, "r", encoding="utf-8") as f:
            input_text = f.read()
        if len(sys.argv) <= 2:
            chapter_title = os.path.splitext(os.path.basename(sys.argv[1]))[0]

    asyncio.run(run_novel2comic(input_text, chapter_title))
