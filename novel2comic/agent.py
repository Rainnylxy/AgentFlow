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
import os
import sys
from datetime import datetime

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
# Prompt 模板：漫画分镜师（PlanExecute 模式）
# ============================================================

class ComicRole(Section):
    """角色定义 + 工作流程"""
    name = "role_card"
    order = 10

    def render(self, context: dict) -> str:
        return (
            "## Role\n"
            "你是专业的漫画分镜师，精通日本漫画(ネーム/Name)与韩式条漫(Webtoon)的分镜设计。\n\n"
            "## 工作流程\n"
            "你需要将一个小说片段转化为完整的分镜脚本，分两步走：\n\n"
            "**第一步：分析 + 规划 (Plan)**\n"
            "1. 分析文本：类型、风格判断(manga/webtoon)、人物列表、情感基调\n"
            "2. 拆分场景：找出关键叙事场景（3-6个），每个场景概括为1-2句话\n"
            "3. 确定每场景的分镜格数（3-6格）\n\n"
            "**第二步：执行生成 (Execute)**\n"
            "按场景逐一生成完整的分镜 Markdown，每格包含：\n"
            "- 画面描述（中文，含前景/中景/背景构图）\n"
            "- 角色动作和表情\n"
            "- 台词（无则留空）\n"
            "- 镜头角度\n"
            "- 情绪氛围\n"
            "- SD 生图 prompt（英文，含画风关键词）\n\n"
            "**第三步：保存**\n"
            "全部生成完毕后，调用 save_final_storyboard 工具保存到文件。"
        )


class ComicStyleGuide(Section):
    """风格规范"""
    name = "style_guide"
    order = 20

    def render(self, context: dict) -> str:
        return (
            "## 漫画风格规范\n\n"
            "### 日式漫画 (manga)\n"
            "- 黑白为主，灰度网点点缀\n"
            "- 视觉动线引导 + 特写/远景交替制造节奏\n"
            "- 速度线、集中线、效果字增强表现力\n"
            "- sd_prompt 关键词: manga style, black and white, screentone, speed lines\n\n"
            "### 彩色条漫 (webtoon)\n"
            "- 全彩色，柔和调色板\n"
            "- 竖屏滑动，每格宽度一致，人物居中偏上\n"
            "- 格间留白控制节奏\n"
            "- sd_prompt 关键词: webtoon style, full color, vertical scroll, soft palette\n\n"
            "### 自动判断规则\n"
            "- 轻小说/校园/恋爱 → manga\n"
            "- 网文/都市/职场 → webtoon\n"
            "- 武侠/玄幻/仙侠 → manga\n"
            "- 科幻/悬疑 → 根据节奏自定"
        )


class ComicOutputFormat(Section):
    """输出格式模板"""
    name = "output_format"
    order = 30

    def render(self, context: dict) -> str:
        return (
            "## 最终输出格式（Markdown）\n\n"
            "生成的分镜脚本必须严格按以下 Markdown 格式输出：\n\n"
            "```\n"
            "# [章节标题]\n\n"
            "## 基础信息\n"
            "- 风格: manga / webtoon\n"
            "- 类型: [小说类型]\n"
            "- 人物: [人物列表]\n"
            "- 基调: [情感基调]\n\n"
            "## 场景1: [场景标题]\n"
            "> 摘要: [1-2句场景概述]\n"
            "> 情绪: [场景情绪]\n\n"
            "### 格1: [简短标题]\n"
            "- **画面**: [中文画面描述 + 构图]\n"
            "- **动作**: [角色动作表情]\n"
            '- **台词**: 「[对话内容]」(无台词则写「无」)\n'
            "- **镜头**: [景别/角度]\n"
            "- **情绪**: [情绪氛围]\n"
            "- **SD Prompt**: `[英文prompt]`\n\n"
            "### 格2: ...\n"
            "...(重复每个场景的每一格)...\n\n"
            "## SD Prompts 汇总\n"
            "1. [第一格prompt]\n"
            "2. [第二格prompt]\n"
            "...\n"
            "```"
        )


class ComicQualityRules(Section):
    """质量约束"""
    name = "quality_rules"
    order = 40

    def render(self, context: dict) -> str:
        return (
            "## 质量规范\n"
            "1. 每场景 3-6 格分镜，动作场景可到 8 格\n"
            "2. 画面描述必须有构图信息（前景/中景/背景）\n"
            "3. SD prompt 必须包含画风关键词 + 关键视觉元素 + 画幅比例(16:9/4:3)\n"
            "4. 关键情感转折台词不能遗漏\n"
            "5. 人物首次出现描述外貌特征，后续用名字指代\n"
            "6. 相邻格之间要有视觉变化（景别切换/视角变化），避免单调\n"
            "7. 分镜脚本用中文，SD prompt 用英文"
        )


def create_prompt_template() -> PromptTemplate:
    template = PromptTemplate("comic_storyboard")
    template.add(ComicRole())
    template.add(ComicStyleGuide())
    template.add(ComicOutputFormat())
    template.add(ComicQualityRules())
    return template


# ============================================================
# 1 个 Tool：保存最终结果
# ============================================================

@tool
def save_final_storyboard(markdown_content: str, chapter_title: str) -> str:
    """将完整的漫画分镜 Markdown 脚本保存到 outputs 目录。

    必须在分镜脚本全部生成完毕后再调用此工具。
    不要在生成过程中调用——全部写完后再保存。

    Args:
        markdown_content: 完整的 Markdown 格式分镜脚本
        chapter_title: 章节标题（用于生成文件名）

    Returns:
        保存的文件路径
    """
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = "".join(c if c.isalnum() or c in "._- " else "_" for c in chapter_title)
    filename = f"{timestamp}_{safe_title}.md"
    path = os.path.join(output_dir, filename)

    with open(path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    # 统计信息
    line_count = markdown_content.count("\n") + 1
    return f"已保存 ({line_count} 行, {len(markdown_content)} 字符)\n文件: {path}"


# ============================================================
# Agent 构建
# ============================================================

def build_agent():
    """构建 Novel2Comic Agent。"""
    api_key = os.getenv("AGENTFLOW_API_KEY", "")
    base_url = os.getenv("AGENTFLOW_BASE_URL", "https://api.deepseek.com/")
    model = os.getenv("AGENTFLOW_MODEL", "deepseek-v4-pro")
    proxy = os.getenv("AGENTFLOW_PROXY", "")

    if not api_key:
        print("[!] AGENTFLOW_API_KEY not set.")
        print("    请设置环境变量: $env:AGENTFLOW_API_KEY='sk-your-key'")
        sys.exit(1)

    llm = OpenAIClient(
        api_key=api_key, model=model, base_url=base_url,
        proxy=proxy or None,
    )
    prompt = create_prompt_template()

    agent = (AgentBuilder("novel2comic")
        .with_llm(llm)
        .with_tools(save_final_storyboard)
        .with_prompt(prompt)
        .with_memory(MemoryProfile.light())       # 不需要复杂记忆
        .with_thinking(ThinkingMode.REACT)  # ReAct 才完整支持 tool 调用循环
        .with_max_iterations(15)
        .build())

    return agent


async def run_novel2comic(text: str, title: str = "未命名章节") -> str:
    """执行小说→漫画分镜转换。"""
    agent = build_agent()

    task = (
        f"## 任务：将以下小说片段转化为漫画分镜脚本\n\n"
        f"章节标题：{title}\n\n"
        f"### 小说原文\n{text}\n\n"
        f"### 执行步骤\n"
        f"**第一步 (Plan)**：分析文本类型/风格/人物/基调，拆分 3-5 个关键场景。\n"
        f"**第二步 (Execute)**：按场景逐一生成完整分镜 Markdown（每格含画面描述、动作、台词、镜头、情绪、SD prompt）。\n"
        f"**第三步 (Save)**：全部分镜写完后，调用 save_final_storyboard(markdown_content=完整markdown, chapter_title=\"{title}\") 保存。\n\n"
        f"注意：在第三步之前不要调用工具——先把分镜内容全部生成完，最后一次性保存。"
    )

    print(f"[novel2comic] 开始处理: {title}")
    print(f"[novel2comic] 文本长度: {len(text)} 字符")
    print(f"[novel2comic] 模式: PlanExecute")

    result = await agent.run(task)

    tool_called = any(
        s.get("type") == "tool_call" for s in result.steps
    )

    print(f"[novel2comic] 处理完成, 步骤数: {len(result.steps)}")
    for i, step in enumerate(result.steps):
        phase = step.get("phase", step.get("type", "?"))
        if phase == "tool_call":
            print(f"  步骤{i}: [TOOL] 调用 → {step.get('calls', [])}")
        else:
            output_preview = step.get("output", "")[:80]
            print(f"  步骤{i}: [{phase}] {output_preview}...")

    if tool_called:
        print(f"[novel2comic] ✅ 工具已被调用，文件由 save_final_storyboard 保存")
    else:
        print(f"[novel2comic] ❌ LLM 未调用工具——工具注册或策略传参可能有问题")

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

    if os.path.isfile(input_text):
        with open(input_text, "r", encoding="utf-8") as f:
            input_text = f.read()
        if len(sys.argv) <= 2:
            chapter_title = os.path.splitext(os.path.basename(sys.argv[1]))[0]

    asyncio.run(run_novel2comic(input_text, chapter_title))
