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

import os as _os

from agentflow.runtime.builder import AgentBuilder
from agentflow.runtime.toolkit import tool
from agentflow.runtime.memory.manager import MemoryProfile
from agentflow.runtime.thinking import ThinkingMode
from agentflow.runtime.llm_client import OpenAIClient

# Skill 文件目录（novel2comic 项目根下的 skills/）
_SKILLS_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "skills")


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

    agent = (AgentBuilder("novel2comic")
        .with_llm(llm)
        .with_skills_dir(_SKILLS_DIR)
        .with_skill("novel2comic")    # Skill 替代硬编码的 Section 类
        .with_tools(save_final_storyboard)
        .with_memory(MemoryProfile.light())
        .with_thinking(ThinkingMode.REACT)
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
