# Novel2Comic Agent 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 基于 AgentFlow 框架，在 AgentFlow 代码库外独立构建 novel2comic Agent 项目

**Architecture:** 单文件 agent.py（4 个 @tool + PromptTemplate + AgentBuilder），配合 requirements.txt + .env.example + example.py

**Tech Stack:** Python 3.10+, AgentFlow (本地路径), DeepSeek API

---

### Task 1: 创建 novel2comic 项目

**Files:**
- Create: `novel2comic/agent.py`
- Create: `novel2comic/requirements.txt`
- Create: `novel2comic/.env.example`
- Create: `novel2comic/example.py`

- [ ] **Step 1: 创建项目目录和 requirements.txt**

```bash
mkdir -p /d/Codes_lxy/VibeCoding/AgentFlow/novel2comic/outputs
```

Create `novel2comic/requirements.txt`:
```
openai>=1.0.0
httpx>=0.27.0
pydantic>=2.0.0
```

AgentFlow 通过本地路径引入（开发阶段），不通过 pip。

- [ ] **Step 2: 创建 .env.example**

```
# DeepSeek API 配置
AGENTFLOW_API_KEY=sk-your-deepseek-key
AGENTFLOW_BASE_URL=https://api.deepseek.com/v1
AGENTFLOW_MODEL=deepseek-chat

# HTTP 代理（可选）
AGENTFLOW_PROXY=http://127.0.0.1:3067
```

- [ ] **Step 3: 写 agent.py（完整实现）**

这是核心文件，包含 4 个 Tool + Prompt 模板 + AgentBuilder。

```python
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
    name = "role_card"
    order = 10

    def render(self, context: dict) -> str:
        return (
            "## Role\n"
            "你是一位专业的漫画分镜师(Comic Storyboard Artist)，精通日本漫画(ネーム/Name)"
            "和韩式条漫(Webtoon)的分镜设计。你的任务是将小说文字转化为可视化的"
            "漫画分镜脚本，为每一格生成画面描述和对应的 Stable Diffusion/DALL-E 生图 prompt。\n\n"
            "## 工作流程\n"
            "1. 先用 analyze_text 分析文本风格、人物、基调\n"
            "2. 再用 extract_scenes 拆分为关键场景\n"
            "3. 然后对每个场景调用 storyboard_scene 生成分镜\n"
            "4. 最后用 compile_chapter 汇总输出"
        )


class ComicStyleGuide(Section):
    name = "style_guide"
    order = 20

    def render(self, context: dict) -> str:
        return (
            "## 漫画风格规范\n"
            "### 日式漫画 (manga)\n"
            "- 黑白为主，可点缀灰度网点和少量高光色\n"
            "- 右翻页阅读顺序\n"
            "- 注重视觉动线引导（读者视线从左到右、从上到下）\n"
            "- 特写与远景交替制造节奏感\n"
            "- 使用速度线、集中线、效果字(拟声词)增强表现力\n"
            "- 对话框融入画面构图，形状随情绪变化\n\n"
            "### 彩色条漫 (webtoon)\n"
            "- 全彩色，柔和的色彩调色板\n"
            "- 竖屏滑动阅读，每格宽度一致\n"
            "- 人物居中偏上，对话框在上方空白处\n"
            "- 格与格之间留白控制阅读节奏\n"
            "- 重点时刻用大格或跨格表现\n\n"
            "### 自动判断 (auto)\n"
            "- 轻小说/校园/恋爱 → manga\n"
            "- 网文/都市/职场 → webtoon\n"
            "- 武侠/玄幻/仙侠 → manga\n"
            "- 科幻/悬疑 → 根据节奏自行判断"
        )


class ComicOutputFormat(Section):
    name = "output_format"
    order = 30

    def render(self, context: dict) -> str:
        return (
            "## 分镜输出格式\n"
            "每个场景的分镜(storyboard_scene)必须按以下格式返回：\n"
            "```json\n"
            "[\n"
            "  {\n"
            '    "panel_number": 1,\n'
            '    "visual_description": "画面构图的中文描述（含前景/中景/背景）",\n'
            '    "character_action": "角色的动作和表情",\n'
            '    "dialogue": "该格的台词内容（无台词则留空）",\n'
            '    "camera_angle": "镜头角度的中文描述（特写/中景/远景/俯视/仰视）",\n'
            '    "mood": "该格的情绪氛围",\n'
            '    "sd_prompt": "英文 Stable Diffusion prompt，包含画风关键词(anime style / manga style / line art / webtoon style)和该格的关键视觉元素"\n'
            "  }\n"
            "]\n"
            "```\n\n"
            "compile_chapter 必须输出 Markdown 格式：\n"
            "- # 章标题\n"
            "- ## 基础信息（风格、人物一览）\n"
            "- ## 场景N: 标题（每场景一个二级标题）\n"
            "- 每个分镜用 ### 格N: 画面描述 的格式\n"
            "- 文末附 ## SD Prompts 汇总"
        )


class ComicQualityRules(Section):
    name = "quality_rules"
    order = 40

    def render(self, context: dict) -> str:
        return (
            "## 质量规范\n"
            "1. 每个场景 3-6 格分镜，动作场景可适当增加\n"
            "2. 画面描述必须有构图感：说清楚前景有什么、中景有什么、背景是什么\n"
            "3. sd_prompt 必须包含画风关键词 + 画幅比例 + 该格独特视觉元素\n"
            "4. 关键对话不能遗漏，尤其是有情感转折的台词\n"
            "5. 人物首次出现时必须描述外貌特征（发型/服饰/体型），后续用名字指代\n"
            "6. 相邻格之间要有视觉变化（景别切换/视角变化），避免单调\n"
            "7. 用 extract_scenes 时，max_scenes 不要超过 8"
        )


def create_prompt_template() -> PromptTemplate:
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
    """分析小说文本的类型、风格、主要人物、情感基调。

    输入完整的小说文本或片段，返回结构化的分析结果。

    返回 JSON 格式:
    {type, style, characters: [{name, role, traits}], tone, era}
    """
    # 这个函数本身是空壳——LLM 只是看到它的签名和描述，
    # 它调用这个 tool 时，LLM 自己会"生成"分析结果。
    # AgentFlow 的 ToolKit 会把调用参数传给 LLM，LLM 在
    # function_call 的 arguments 里写好参数，然后我们在这里
    # 可以实际执行。但对于纯文本分析类 tool，我们直接返回
    # "请根据你的分析能力返回结果"即可，LLM 会在下一轮给出分析。
    return json.dumps({
        "status": "analyzed",
        "instruction": "Based on the text provided, output your analysis as a structured JSON object with fields: type, style, characters (list of {name, role, traits}), tone, era"
    }, ensure_ascii=False)


@tool
def extract_scenes(text: str, max_scenes: int = 8) -> str:
    """将小说文本拆分为关键场景列表。

    每个场景是一个独立的叙事单元，有明确的起承转合。

    返回 JSON 数组:
    [{id, title, summary, characters_involved, emotion, key_dialogue}]
    """
    return json.dumps({
        "status": "scenes_extracted",
        "instruction": f"Extract up to {max_scenes} key scenes. For each scene provide: id, title, summary (1 sentence), characters_involved (list of names), emotion, key_dialogue (most important line). Output as JSON array."
    }, ensure_ascii=False)


@tool
def storyboard_scene(
    scene_summary: str,
    characters: str,  # JSON string of character list
    style: str = "auto",
    panels_per_scene: int = 4
) -> str:
    """为一个场景生成漫画分镜。

    根据场景摘要和人物信息，生成该场景的完整分镜脚本。

    Args:
        scene_summary: 场景摘要描述
        characters: JSON 格式的人物列表 [{name, role, traits}, ...]
        style: 漫画风格 manga/webtoon/auto
        panels_per_scene: 该场景的分镜格数（建议 3-6）

    返回 JSON 数组:
    [{panel_number, visual_description, character_action, dialogue, camera_angle, mood, sd_prompt}]
    """
    return json.dumps({
        "status": "storyboard_ready",
        "instruction": (
            f"Generate {panels_per_scene} panels for this scene in {style} style. "
            f"For each panel provide: panel_number, visual_description (Chinese, with composition details), "
            f"character_action, dialogue, camera_angle (Chinese), mood, sd_prompt (English, include anime/manga/webtoon style keywords). "
            f"Use the characters: {characters}. Output as JSON array."
        )
    }, ensure_ascii=False)


@tool
def compile_chapter(
    chapter_title: str,
    scenes_storyboard: str,  # JSON string of all scenes
    style: str = "auto"
) -> str:
    """将全部分镜汇总为 Markdown 格式输出。

    汇总所有场景的分镜，生成最终的漫画分镜脚本文件。

    Args:
        chapter_title: 章节标题
        scenes_storyboard: JSON 格式的全部分镜数据
        style: 漫画风格

    返回完整的 Markdown 格式分镜脚本
    """
    return json.dumps({
        "status": "compiled",
        "instruction": (
            f"Compile all storyboards into a Markdown document. Title: '{chapter_title}', Style: {style}. "
            f"Include: # chapter title, ## character list, ## each scene with ### per panel, "
            f"## SD Prompts summary at the end with all prompts listed. "
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
        print("[!] AGENTFLOW_API_KEY not set. 请设置环境变量或创建 .env 文件。")
        print("    export AGENTFLOW_API_KEY='sk-your-key'")
        sys.exit(1)

    llm = OpenAIClient(api_key=api_key, model=model, base_url=base_url, proxy=proxy or None)
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

    # 构建引导指令
    task = (
        f"## 章节标题\n{title}\n\n"
        f"## 小说原文\n{text}\n\n"
        f"## 任务\n"
        f"请按以下步骤执行：\n"
        f"1. 调用 analyze_text 分析文本的类型、风格、人物、基调\n"
        f"2. 调用 extract_scenes 拆分关键场景\n"
        f"3. 对每个场景调用 storyboard_scene 生成分镜\n"
        f"4. 调用 compile_chapter 汇总为完整的分镜脚本\n\n"
        f"注意：风格默认为 auto（自动判断），如果要指定风格请告知。"
    )

    print(f"[novel2comic] 开始处理: {title}")
    print(f"[novel2comic] 文本长度: {len(text)} 字符")
    print()

    result = await agent.run(task)

    # 保存输出
    os.makedirs("outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"outputs/{timestamp}_{title}.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# Novel2Comic 分镜脚本\n\n")
        f.write(f"- 章节: {title}\n")
        f.write(f"- 生成时间: {datetime.now().isoformat()}\n")
        f.write(f"- 文本长度: {len(text)} 字符\n\n")
        f.write("---\n\n")
        f.write(result.output)

    print(f"[novel2comic] 输出已保存: {output_file}")
    print(f"[novel2comic] Agent 执行步骤: {len(result.steps)}")
    for i, step in enumerate(result.steps):
        step_type = step.get("type", step.get("phase", "?"))
        print(f"    步骤 {i}: {step_type}")

    return result.output


# ============================================================
# CLI 入口
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python agent.py <小说文本或文件路径> [章节标题]")
        print()
        print("示例:")
        print('  python agent.py "一个少年在月光下拔出了剑..." 第一章')
        print("  python agent.py chapter1.txt 月下初遇")
        sys.exit(1)

    input_text = sys.argv[1]
    chapter_title = sys.argv[2] if len(sys.argv) > 2 else "未命名章节"

    # 如果是文件路径，读文件内容
    if os.path.isfile(input_text):
        with open(input_text, "r", encoding="utf-8") as f:
            input_text = f.read()
        if len(sys.argv) <= 2:
            chapter_title = os.path.splitext(os.path.basename(sys.argv[1]))[0]

    asyncio.run(run_novel2comic(input_text, chapter_title))
```

- [ ] **Step 4: 创建 example.py**

```python
# -*- coding: utf-8 -*-
"""
Novel2Comic 示例脚本

用法:
    python example.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from novel2comic.agent import build_agent


async def main():
    # 示例小说片段
    sample_text = """
    夜幕降临，长安城华灯初上。苏墨站在朱雀大街的尽头，手握一柄锈迹斑斑的铁剑。

    "三年了，我终于回来了。"他低声自语，目光穿过熙攘的人潮，锁定在那座金碧辉煌的将军府上。

    一个卖糖葫芦的老者经过，苏墨叫住了他："老人家，将军府近日可有什么动静？"

    老者打量了他一眼，压低声音道："小兄弟，你怕是外地来的吧？将军府三日前贴出告示，要招纳天下剑客，说是要缉拿一个叫'夜枭'的大盗。赏金一千两黄金。"

    "一千两黄金..."苏墨嘴角微扬，眼中闪过一丝复杂的神色。

    他绕过朱雀大街，钻进一条暗巷。一只黑猫从墙头跃下，落在他肩上。苏墨从怀中取出一张泛黄的羊皮纸，上面画着将军府的内部地形图。

    "夜枭...呵，他们连我的真名都不知道了。"他收起羊皮纸，身形一闪，消失在夜色中。
    """

    agent = build_agent()
    print("Novel2Comic Agent 已就绪")
    print(f"示例文本: {len(sample_text)} 字符\n")

    result = await agent.run(
        f"## 章节标题\n月下归来\n\n"
        f"## 小说原文\n{sample_text}\n\n"
        f"## 任务\n请按以下步骤执行：\n"
        f"1. 调用 analyze_text 分析文本\n"
        f"2. 调用 extract_scenes 拆分场景\n"
        f"3. 对每个场景调用 storyboard_scene 生成分镜\n"
        f"4. 调用 compile_chapter 汇总输出\n"
    )

    print("=" * 60)
    print("生成的分镜脚本:")
    print("=" * 60)
    print(result.output)
    print()
    print(f"Agent 执行步骤数: {len(result.steps)}")
    for i, step in enumerate(result.steps):
        print(f"  步骤{i}: {step.get('type', step.get('phase', '?'))}")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 5: 验证 Agent 可以构建**

```bash
cd /d/Codes_lxy/VibeCoding/AgentFlow && python -c "
import sys; sys.path.insert(0, '.')
from novel2comic.agent import build_agent
print('Agent build OK')
"
```

Expected: 无报错（如果 API key 未设置会提示退出，正常）
Expected with API key set: 构建成功

- [ ] **Step 6: 提交**

```bash
git add novel2comic/
git commit -m "feat: add novel2comic agent project"
```
