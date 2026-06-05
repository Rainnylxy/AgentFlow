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
from typing import Optional

# Windows 修复：asyncio 事件循环提前关闭问题
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

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
            "## 重要规则：必须使用工具！\n"
            "你生成的分镜内容必须通过调用工具函数来保存，不要直接在文字中输出 JSON。\n"
            "你必须在同一轮回复中调用工具（function_call），不能只描述要做什么。\n\n"
            "## 工作流程（严格按顺序执行工具调用）\n"
            "1. 分析文本类型、风格、人物 → 调用 save_analysis(analysis_json) 保存\n"
            "2. 拆分关键场景 → 调用 save_scenes(scenes_json) 保存\n"
            "3. 对每个场景生成分镜 → 逐个调用 save_storyboard(scene_id, panels_json) 保存\n"
            "4. 所有场景完成 → 调用 compile_final_output(chapter_title, style) 汇总\n\n"
            "每完成一步工具调用，等待结果返回后再进行下一步。"
            "不要在文本回复中直接输出分镜 JSON——必须通过工具函数保存。"
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
# 工作目录（Tool 做真实 I/O 时使用）
# ============================================================

_WORK_DIR: Optional[str] = None


def _get_work_dir() -> str:
    global _WORK_DIR
    if _WORK_DIR is None:
        _WORK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(_WORK_DIR, exist_ok=True)
    return _WORK_DIR


def _set_work_dir(path: str) -> None:
    global _WORK_DIR
    _WORK_DIR = path
    os.makedirs(_WORK_DIR, exist_ok=True)


# ============================================================
# 4 个 Tool —— 每个都做真实 I/O 工作，不是返回指令
# ============================================================

_VALID_STYLES = {"manga", "webtoon", "auto"}


@tool
def save_analysis(analysis_json: str) -> str:
    """保存文本分析结果到工作目录。

    分析结果必须是 JSON 格式，包含：
    - type: 小说类型
    - recommended_style: 推荐漫画风格（manga 或 webtoon）
    - characters: 人物列表 [{name, role, traits}]
    - tone: 整体基调
    - era: 时代背景

    Args:
        analysis_json: JSON 格式的分析结果字符串
    """
    try:
        data = json.loads(analysis_json)
        style = data.get("recommended_style", "auto")
        if style not in _VALID_STYLES:
            return f"[错误] recommended_style 必须是 manga/webtoon/auto，收到: {style}"

        work_dir = _get_work_dir()
        path = os.path.join(work_dir, "_analysis.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        chars = data.get("characters", [])
        char_names = [c.get("name", "?") for c in chars]
        return (
            f"✅ 分析结果已保存 ({len(char_names)} 个人物: {', '.join(char_names)})\n"
            f"   类型: {data.get('type', '?')}\n"
            f"   风格: {style}\n"
            f"   基调: {data.get('tone', '?')}\n"
            f"   文件: {path}"
        )
    except json.JSONDecodeError as e:
        return f"[错误] analysis_json 不是有效的 JSON: {e}"


@tool
def save_scenes(scenes_json: str) -> str:
    """保存场景拆分结果到工作目录。

    必须是 JSON 数组，每个元素包含：
    {id, title, summary, characters_involved, emotion, key_dialogue}

    Args:
        scenes_json: JSON 格式的场景数组字符串
    """
    try:
        scenes = json.loads(scenes_json)
        if not isinstance(scenes, list):
            return f"[错误] scenes_json 必须是 JSON 数组，收到: {type(scenes).__name__}"
        if len(scenes) == 0:
            return "[错误] 场景列表为空，请至少提取 1 个场景"
        if len(scenes) > 8:
            return f"[错误] 场景数 ({len(scenes)}) 超过上限 (8)，请合并或减少场景"

        work_dir = _get_work_dir()
        path = os.path.join(work_dir, "_scenes.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(scenes, f, ensure_ascii=False, indent=2)

        titles = [s.get("title", f"场景{s.get('id', '?')}") for s in scenes]
        return (
            f"✅ 已保存 {len(scenes)} 个场景:\n"
            + "\n".join(f"   {i+1}. {t}" for i, t in enumerate(titles))
            + f"\n   文件: {path}"
        )
    except json.JSONDecodeError as e:
        return f"[错误] scenes_json 不是有效的 JSON: {e}"


@tool
def save_storyboard(scene_id: int, panels_json: str) -> str:
    """保存单个场景的漫画分镜到工作目录。

    panels_json 必须是 JSON 数组，每格包含：
    {panel_number, visual_description, character_action, dialogue, camera_angle, mood, sd_prompt}

    Args:
        scene_id: 场景编号（从 1 开始）
        panels_json: JSON 格式的分镜面板数组字符串
    """
    try:
        panels = json.loads(panels_json)
        if not isinstance(panels, list):
            return f"[错误] panels_json 必须是 JSON 数组，收到: {type(panels).__name__}"
        if len(panels) == 0:
            return f"[错误] 场景 {scene_id} 的分镜面板数为 0，每场景至少需要 1 格"
        if len(panels) > 10:
            return f"[错误] 场景 {scene_id} 的分镜面板数 ({len(panels)}) 过多，建议 3-6 格"

        # 校验每格必填字段
        required_fields = ["panel_number", "visual_description", "dialogue", "sd_prompt"]
        for p in panels:
            missing = [f for f in required_fields if f not in p]
            if missing:
                return f"[错误] 场景 {scene_id} 第 {p.get('panel_number', '?')} 格缺少字段: {missing}"

        work_dir = _get_work_dir()
        path = os.path.join(work_dir, f"scene_{scene_id:02d}_storyboard.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(panels, f, ensure_ascii=False, indent=2)

        return (
            f"✅ 场景 {scene_id} 分镜已保存 ({len(panels)} 格)\n"
            f"   文件: {path}"
        )
    except json.JSONDecodeError as e:
        return f"[错误] panels_json 不是有效的 JSON: {e}"


@tool
def compile_final_output(chapter_title: str, style: str = "auto") -> str:
    """读取工作目录中所有已保存的分析、场景、分镜文件，汇总为最终的 Markdown 分镜脚本。

    必须按顺序执行：save_analysis → save_scenes → save_storyboard(每个场景) → compile_final_output。
    如果找不到中间文件，会返回错误。

    Args:
        chapter_title: 章节标题
        style: 漫画风格 (manga/webtoon/auto)

    返回最终 Markdown 文件路径。
    """
    if style not in _VALID_STYLES:
        return f"[错误] style 必须是 manga/webtoon/auto，收到: {style}"

    work_dir = _get_work_dir()

    # 读取分析结果
    analysis_path = os.path.join(work_dir, "_analysis.json")
    if not os.path.exists(analysis_path):
        return f"[错误] 未找到分析文件 {analysis_path}，请先调用 save_analysis"
    with open(analysis_path, "r", encoding="utf-8") as f:
        analysis = json.load(f)

    # 读取场景列表
    scenes_path = os.path.join(work_dir, "_scenes.json")
    if not os.path.exists(scenes_path):
        return f"[错误] 未找到场景文件 {scenes_path}，请先调用 save_scenes"
    with open(scenes_path, "r", encoding="utf-8") as f:
        scenes = json.load(f)

    # 读取所有场景分镜文件
    storyboards = []
    for scene in scenes:
        sid = scene.get("id", len(storyboards) + 1)
        sb_path = os.path.join(work_dir, f"scene_{int(sid):02d}_storyboard.json")
        if os.path.exists(sb_path):
            with open(sb_path, "r", encoding="utf-8") as f:
                storyboards.append({"scene": scene, "panels": json.load(f)})
        else:
            storyboards.append({"scene": scene, "panels": [], "missing": True})

    # 组装 Markdown
    now = datetime.now()
    lines = []
    lines.append(f"# {chapter_title}")
    lines.append("")
    lines.append("## 基础信息")
    lines.append("")
    lines.append(f"- **章节**: {chapter_title}")
    lines.append(f"- **风格**: {style}")
    lines.append(f"- **类型**: {analysis.get('type', '?')}")
    lines.append(f"- **基调**: {analysis.get('tone', '?')}")
    lines.append(f"- **时代**: {analysis.get('era', '?')}")
    lines.append(f"- **生成时间**: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # 人物表
    characters = analysis.get("characters", [])
    if characters:
        lines.append("## 人物一览")
        lines.append("")
        lines.append("| 姓名 | 身份 | 特征 |")
        lines.append("|------|------|------|")
        for c in characters:
            name = c.get("name", "?")
            role = c.get("role", "?")
            traits = ", ".join(c.get("traits", [])) if isinstance(c.get("traits"), list) else c.get("traits", "?")
            lines.append(f"| {name} | {role} | {traits} |")
        lines.append("")

    # 每个场景的分镜
    all_prompts = []
    for idx, sb in enumerate(storyboards, 1):
        scene = sb["scene"]
        panels = sb["panels"]
        lines.append(f"## 场景{idx}: {scene.get('title', f'场景{idx}')}")
        lines.append("")
        lines.append(f"> **摘要**: {scene.get('summary', '')}")
        lines.append(f"> **情绪**: {scene.get('emotion', '')}")
        if scene.get("key_dialogue"):
            lines.append(f"> **关键台词**: 「{scene['key_dialogue']}」")
        lines.append("")

        if sb.get("missing"):
            lines.append("⚠️ 该场景的分镜文件缺失")
            lines.append("")
            continue

        for p in panels:
            pn = p.get("panel_number", "?")
            lines.append(f"### 格{pn}: {p.get('visual_description', '')[:60]}...")
            lines.append("")
            lines.append(f"- **画面描述**: {p.get('visual_description', '')}")
            lines.append(f"- **角色动作**: {p.get('character_action', '')}")
            if p.get("dialogue"):
                lines.append(f"- **台词**: 「{p['dialogue']}」")
            lines.append(f"- **镜头**: {p.get('camera_angle', '')}")
            lines.append(f"- **情绪**: {p.get('mood', '')}")
            lines.append(f"- **SD Prompt**: `{p.get('sd_prompt', '')}`")
            lines.append("")

            if p.get("sd_prompt"):
                all_prompts.append(f"格{pn}: {p['sd_prompt']}")

    # SD Prompts 汇总
    if all_prompts:
        lines.append("## SD Prompts 汇总")
        lines.append("")
        for i, prompt in enumerate(all_prompts, 1):
            lines.append(f"{i}. {prompt}")
        lines.append("")

    # 写入最终文件
    safe_title = "".join(c if c.isalnum() or c in "._- " else "_" for c in chapter_title)
    output_path = os.path.join(work_dir, f"{now.strftime('%Y%m%d_%H%M%S')}_{safe_title}.md")
    content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    total_panels = sum(len(sb["panels"]) for sb in storyboards if not sb.get("missing"))
    missing_count = sum(1 for sb in storyboards if sb.get("missing"))
    return (
        f"✅ 最终分镜脚本已生成\n"
        f"   文件: {output_path}\n"
        f"   场景数: {len(storyboards)} (其中 {missing_count} 个缺失分镜)\n"
        f"   总格数: {total_panels}\n"
        f"   SD Prompts: {len(all_prompts)} 个"
    )


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
        .with_tools(save_analysis, save_scenes, save_storyboard, compile_final_output)
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
        f"1. 分析文本，将结果作为 JSON 字符串传给 save_analysis 保存\n"
        f"2. 拆分关键场景，将场景数组作为 JSON 字符串传给 save_scenes 保存\n"
        f"3. 对每个场景依次生成分镜，将每个场景的分镜数组传给 save_storyboard 保存\n"
        f"   （scene_id 从 1 开始递增，panels_json 为各格分镜的 JSON 数组）\n"
        f"4. 所有场景处理完毕后，调用 compile_final_output(chapter_title=\"{title}\")\n"
        f"   汇总为最终的 Markdown 分镜脚本\n\n"
        f"注意：风格默认为 auto（自动判断），根据分析结果确定。"
    )

    print(f"[novel2comic] 开始处理: {title}")
    print(f"[novel2comic] 文本长度: {len(text)} 字符")

    result = await agent.run(task)

    print(f"[novel2comic] 处理完成")
    print(f"[novel2comic] Agent 执行步骤数: {len(result.steps)}")
    for i, step in enumerate(result.steps):
        step_type = step.get("type", step.get("phase", "?"))
        if step_type == "tool_call":
            calls = step.get("calls", [])
            print(f"    步骤{i}: 调用工具 → {', '.join(calls)}")
        elif step_type == "final":
            print(f"    步骤{i}: 最终输出")

    print()
    print(result.output[:500] + "..." if len(result.output) > 500 else result.output)

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
