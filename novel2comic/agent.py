# -*- coding: utf-8 -*-
"""
Novel2Comic Agent V2
====================
Agent 驱动的"小说→漫画"生成系统。

使用 AgentFlow 框架：Skill + ToolKit + Memory + Thinking。
Agent 自主决策调用 6 个 Pipeline Tool，用户通过自然语言交互。

用法:
    python agent.py "小说文本内容"
    python agent.py chapter1.txt --title "月下归来"
    python agent.py --load projects/xxx/chapter_data.json
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Optional

# Windows 修复
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 添加路径
# novel2comic/ → 用于 from src.models / from src.styles / from src.img_adapter
_n2c_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _n2c_dir)
# 项目根 → 用于 from agentflow.runtime...
_project_root = os.path.dirname(_n2c_dir)
sys.path.insert(0, _project_root)

from agentflow.runtime.builder import AgentBuilder
from agentflow.runtime.toolkit import tool
from agentflow.runtime.memory.manager import MemoryProfile
from agentflow.runtime.thinking import ThinkingMode
from agentflow.runtime.llm_client import OpenAIClient

from src.models import (
    ChapterData, AnalysisResult, CharacterSheet, CharacterAppearance,
    Scene, Panel, ComicPage, StyleProfile, Novel, ChapterInfo,
)
from src.styles import detect_style, BUILTIN_STYLES
from src.img_adapter import ImageGenAdapter
from src.chapter_parser import parse_novel_chapters
from src.novel_registry import (
    register_novel, find_novel, list_all_novels,
    update_novel_access, update_novel_style, update_novel_chapters,
)
from src.novel_rag import (
    NovelRAG, build_context_from_search, build_character_context,
)

# ============================================================
# 共享上下文（Tool 通过此访问 LLM / ImageGen / Data）
# ============================================================

class AgentContext:
    """Tool 共享状态——在 Agent 启动前注入。"""
    def __init__(self):
        self.novel: Optional[Novel] = None         # 全书数据（章节列表 + 角色库）
        self.chapter_data: Optional[ChapterData] = None  # 当前章的 Pipeline 状态
        self.rag: Optional[NovelRAG] = None        # 全书 RAG 知识库
        self.openai_client = None   # openai.OpenAI 同步客户端（供 Tool 内 LLM 调用）
        self.llm_model: str = ""
        self.img_gen: Optional[ImageGenAdapter] = None

    @property
    def data(self) -> Optional[ChapterData]:
        """快捷访问当前章数据。"""
        return self.chapter_data

_ctx = AgentContext()


def _llm_chat_json(system_prompt: str, user_prompt: str, temperature: float = 0.3) -> dict:
    """Tool 内部使用的 LLM JSON 调用。"""
    full_system = system_prompt + "\n\nYou MUST respond with valid JSON only. No markdown fences, no explanation."
    response = _ctx.openai_client.chat.completions.create(
        model=_ctx.llm_model,
        messages=[
            {"role": "system", "content": full_system},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        timeout=120,
        max_tokens=4096,
    )
    text = response.choices[0].message.content or ""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return json.loads(text)


# ============================================================
# Novel 级 Tool（全书管理）
# ============================================================

@tool
def load_novel(file_path: str) -> str:
    """加载一本小说文件。

    首次加载时解析章节并缓存。再次加载同一文件时直接从缓存恢复，
    无需重新解析。支持 .txt 格式。

    Args:
        file_path: 小说 .txt 文件的路径
    """
    if not os.path.isfile(file_path):
        return json.dumps({"error": f"文件不存在: {file_path}"})

    # 1. 检查注册表 —— 是否已解析过
    cached = find_novel(file_path)
    if cached:
        # 缓存命中！直接从 novel.json 恢复
        novel_json_path = os.path.join(cached.project_dir, "novel.json")
        if os.path.exists(novel_json_path):
            _ctx.novel = Novel.load(novel_json_path)
            _ctx.novel.output_dir = cached.project_dir
            update_novel_access(file_path)

            # 加载 RAG 索引
            _ctx.rag = NovelRAG(cached.project_dir)
            rag_path = os.path.join(cached.project_dir, "novel_rag.json")
            if _ctx.rag.load(rag_path):
                print(f"  [RAG] 从缓存加载: {_ctx.rag.total_chunks} 个文本块")

            return json.dumps({
                "status": "ok",
                "cached": True,
                "title": cached.title,
                "total_chapters": cached.total_chapters,
                "style": cached.style,
                "project_dir": cached.project_dir,
                "message": (
                    f"[缓存命中]《{cached.title}》已恢复，共 {cached.total_chapters} 章。"
                    + (f" 全书风格: {cached.style}。" if cached.style else "")
                    + f" 请调用 list_chapters 查看目录，select_chapter(N) 选择章节。"
                ),
            }, ensure_ascii=False)

    # 2. 缓存未命中 —— 解析小说
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    chapters = parse_novel_chapters(text, base_name)

    # 创建项目目录
    project_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "projects", datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(project_dir, exist_ok=True)

    _ctx.novel = Novel(
        title=base_name,
        file_path=file_path,
        chapters=chapters,
        output_dir=project_dir,
    )

    # 持久化
    novel_path = os.path.join(project_dir, "novel.json")
    _ctx.novel.save(novel_path)

    # 构建 RAG 索引
    _ctx.rag = NovelRAG(project_dir)
    rag_path = os.path.join(project_dir, "novel_rag.json")

    # 检查是否有缓存的 RAG
    if not _ctx.rag.load(rag_path):
        print(f"  [RAG] 正在索引 {len(chapters)} 章...")
        chunk_count = _ctx.rag.index_novel(chapters, _ctx.openai_client)
        _ctx.rag.save(rag_path)
        print(f"  [RAG] 索引完成: {chunk_count} 个文本块")
    else:
        print(f"  [RAG] 从缓存加载: {_ctx.rag.total_chunks} 个文本块")

    # 注册到注册表
    register_novel(file_path, base_name, len(chapters), project_dir)

    ch_list = [f"第{ch.index}章: {ch.title} ({ch.word_count}字)" for ch in chapters[:20]]
    preview = "\n".join(ch_list)
    if len(chapters) > 20:
        preview += f"\n... 共 {len(chapters)} 章"

    return json.dumps({
        "status": "ok",
        "cached": False,
        "title": base_name,
        "total_chapters": len(chapters),
        "chapters": ch_list,
        "message": f"[首次解析]《{base_name}》已加载并缓存，共 {len(chapters)} 章。下次访问将直接恢复。请调用 list_chapters 查看目录，select_chapter(N) 选择章节。\n\n{preview}",
    }, ensure_ascii=False)


@tool
def list_novels() -> str:
    """列出所有已加载过的小说（支持从注册表恢复）。

    显示每本小说的标题、章节数、风格和最后访问时间。
    无需先调用 load_novel。
    """
    entries = list_all_novels()

    if not entries:
        return json.dumps({
            "status": "ok",
            "novels": [],
            "message": "还没有加载过任何小说。请用 load_novel(文件路径) 加载一本。",
        }, ensure_ascii=False)

    novel_list = []
    for e in entries:
        novel_list.append(
            f"《{e.title}》({e.total_chapters}章) | 风格: {e.style or '未设置'} | 最后访问: {e.last_accessed[:19] if e.last_accessed else '未知'}"
        )

    return json.dumps({
        "status": "ok",
        "count": len(entries),
        "novels": novel_list,
        "message": (
            f"共 {len(entries)} 本已加载的小说。"
            f" 用 resume_novel({len(entries)} 本中的序号从 0 开始) 恢复，"
            f"或用 load_novel(路径) 加载新的。"
        ),
    }, ensure_ascii=False)


@tool
def resume_novel(novel_index: int = 0) -> str:
    """恢复之前加载过的小说。

    从注册表中按索引恢复，自动加载所有章节和已设计的角色。

    Args:
        novel_index: 小说在列表中的索引（从 0 开始）。调用 list_novels 查看。
    """
    entries = list_all_novels()

    if not entries:
        return json.dumps({"error": "还没有加载过任何小说。请用 load_novel(路径) 加载。"})

    if novel_index < 0 or novel_index >= len(entries):
        return json.dumps({
            "error": f"索引 {novel_index} 无效。可用范围: 0-{len(entries)-1}",
            "available": [f"[{i}]《{e.title}》" for i, e in enumerate(entries)],
        })

    entry = entries[novel_index]

    # 从 novel.json 恢复
    novel_json_path = os.path.join(entry.project_dir, "novel.json")
    if not os.path.exists(novel_json_path):
        return json.dumps({"error": f"小说数据文件不存在: {novel_json_path}。请重新 load_novel('{entry.novel_path}')"})

    _ctx.novel = Novel.load(novel_json_path)
    _ctx.novel.output_dir = entry.project_dir
    update_novel_access(entry.novel_path)

    # 加载 RAG
    _ctx.rag = NovelRAG(entry.project_dir)
    rag_path = os.path.join(entry.project_dir, "novel_rag.json")
    _ctx.rag.load(rag_path)  # 静默加载，没有也不报错

    # 列出章节和角色状态
    completed = sum(1 for ch in _ctx.novel.chapters if ch.status == "completed")
    chars_known = [c.name for c in _ctx.novel.characters]

    return json.dumps({
        "status": "ok",
        "title": entry.title,
        "total_chapters": entry.total_chapters,
        "completed_chapters": completed,
        "style": entry.style,
        "characters_known": chars_known,
        "message": (
            f"已恢复《{entry.title}》，共 {entry.total_chapters} 章"
            f"（已完成 {completed} 章）。"
            + (f" 全书角色: {', '.join(chars_known)}。" if chars_known else "")
            + f" 请调用 list_chapters 查看详情，select_chapter(N) 选择要生成的章节。"
        ),
    }, ensure_ascii=False)


@tool
def list_chapters() -> str:
    """列出当前小说的所有章节及其状态。"""
    novel = _ctx.novel
    if not novel:
        return json.dumps({"error": "请先调用 load_novel 加载小说"})

    lines = []
    for ch in novel.chapters:
        status_icon = "[OK]" if ch.status == "completed" else ("[*]" if ch.status == "generating" else "[ ]")
        lines.append(f"{status_icon} 第{ch.index}章: {ch.title} ({ch.word_count}字)")

    return json.dumps({
        "status": "ok",
        "total": novel.total_chapters,
        "current": novel.current_chapter_index,
        "chapter_list": lines,
        "characters_known": [c.name for c in novel.characters],
        "message": f"当前选中: 第{novel.current_chapter_index}章。用 select_chapter(N) 切换章节。已发现角色: {', '.join(c.name for c in novel.characters) if novel.characters else '（无）'}",
    }, ensure_ascii=False)


@tool
def select_chapter(chapter_index: int) -> str:
    """选择要生成漫画的章节。

    选中后，后续的 analyze_text / design_characters 等工具将针对该章执行。
    之前章节已设计的角色会自动复用。

    Args:
        chapter_index: 章节编号 (1-based)
    """
    novel = _ctx.novel
    if not novel:
        return json.dumps({"error": "请先调用 load_novel 加载小说"})

    chapter = None
    for ch in novel.chapters:
        if ch.index == chapter_index:
            chapter = ch
            break

    if not chapter:
        return json.dumps({"error": f"第{chapter_index}章不存在。可用章节: 1-{novel.total_chapters}"})

    novel.current_chapter_index = chapter_index
    chapter.status = "generating"

    # 为该章创建 ChapterData，继承全书角色库和风格
    ch_output_dir = os.path.join(novel.output_dir, f"chapter_{chapter_index:04d}")
    os.makedirs(ch_output_dir, exist_ok=True)

    _ctx.chapter_data = ChapterData(
        title=f"第{chapter_index}章 {chapter.title}",
        source_text=chapter.content,
        output_dir=ch_output_dir,
        created_at=datetime.now().isoformat(),
    )

    # 继承全书角色库
    _ctx.chapter_data.characters = list(novel.characters)

    # 继承全书风格
    if novel.style_profile:
        _ctx.chapter_data.style_profile = novel.style_profile

    return json.dumps({
        "status": "ok",
        "chapter_index": chapter_index,
        "title": chapter.title,
        "word_count": chapter.word_count,
        "inherited_characters": [c.name for c in novel.characters],
        "inherited_style": novel.style_profile.name if novel.style_profile else "auto",
        "message": (
            f"已选中 第{chapter_index}章《{chapter.title}》({chapter.word_count}字)。"
            + (f" 已从前面章节继承 {len(novel.characters)} 个角色。" if novel.characters else "")
            + " 请调用 analyze_text 开始生成。"
        ),
    }, ensure_ascii=False)


# ============================================================
# RAG 检索 Tool
# ============================================================

@tool
def search_novel(query: str, top_k: int = 5) -> str:
    """在全书内容中检索与查询相关的段落（RAG 语义搜索）。

    可用于查找角色背景、世界观设定、前情提要、特定场景描述等。
    在角色设计或分镜生成之前调用，获取更丰富的上下文。

    Args:
        query: 搜索查询（如 "苏墨的外貌"、"将军府布局"、"暗巷描写"）
        top_k: 返回结果数量，默认 5
    """
    rag = _ctx.rag
    if not rag or not rag.is_indexed:
        return json.dumps({"error": "RAG 索引未就绪。请先 load_novel 加载小说。"})

    results = rag.search(query, top_k=top_k, openai_client=_ctx.openai_client)

    if not results:
        return json.dumps({
            "status": "ok",
            "query": query,
            "results": [],
            "message": "未找到相关内容。",
        }, ensure_ascii=False)

    formatted = []
    for r in results:
        formatted.append({
            "chapter": f"第{r['chapter_index']}章《{r['chapter_title']}》",
            "score": r["score"],
            "text": r["text"][:300],
        })

    return json.dumps({
        "status": "ok",
        "query": query,
        "count": len(formatted),
        "results": formatted,
        "message": f"找到 {len(formatted)} 条相关段落。",
        "context": build_context_from_search(rag, query, _ctx.openai_client, top_k=top_k),
    }, ensure_ascii=False)


@tool
def get_character_info(character_name: str) -> str:
    """获取指定角色在全书中的详细信息。

    检索角色的所有出场场景、外貌描写、对话和互动，
    用于设计角色外貌或编写分镜时保持一致性。

    Args:
        character_name: 角色中文名（如 "苏墨"）
    """
    rag = _ctx.rag
    if not rag or not rag.is_indexed:
        return json.dumps({"error": "RAG 索引未就绪。请先 load_novel 加载小说。"})

    # 1. 用 RAG 搜索角色出场记录
    appearances = rag.search_by_character(character_name, top_k=15)

    # 2. 用嵌入搜索外貌相关描述
    desc_results = rag.search(
        f"{character_name} 外貌 长相 穿着 服饰 特征",
        top_k=5, openai_client=_ctx.openai_client,
    )

    # 3. 搜索角色关系
    relation_results = rag.search(
        f"{character_name} 关系 对话 互动 冲突",
        top_k=5, openai_client=_ctx.openai_client,
    )

    context = build_character_context(rag, character_name)

    return json.dumps({
        "status": "ok",
        "character": character_name,
        "appearance_count": len(appearances),
        "appearances": [
            {"chapter": f"第{r['chapter_index']}章", "text": r["text"][:200]}
            for r in appearances[:8]
        ],
        "descriptions": [
            {"chapter": f"第{r['chapter_index']}章", "score": r["score"], "text": r["text"][:200]}
            for r in desc_results
        ],
        "relations": [
            {"chapter": f"第{r['chapter_index']}章", "score": r["score"], "text": r["text"][:200]}
            for r in relation_results
        ],
        "context": context,
        "message": f"找到角色 '{character_name}' 的 {len(appearances)} 处出场记录。",
    }, ensure_ascii=False)


# ============================================================
# Pipeline Tool（单章生成——共 7 个）
# ============================================================

@tool
def analyze_text(text: str) -> str:
    """分析小说文本：识别题材标签、漫画风格、人物预览、情感基调和时代背景。

    这是 Pipeline 的第一步。调用后会自动判断使用哪种漫画风格 (manga/webtoon/gufeng)。

    Args:
        text: 小说章节的完整文本（或前 3000 字符）
    """
    system_prompt = """你是一位资深的小说编辑和漫画改编顾问。
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

    sample = text[:3000]
    result = _llm_chat_json(system_prompt, f"请分析以下小说片段：\n\n{sample}")

    data = _ctx.data
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
    data.style_profile = detected

    chars = [c["name"] for c in data.analysis.characters_preview]
    return json.dumps({
        "status": "ok",
        "style": detected.name,
        "genre_tags": data.analysis.genre_tags,
        "tone": data.analysis.tone,
        "era": data.analysis.era,
        "pace": data.analysis.pace,
        "characters_found": chars,
        "message": f"分析完成。风格={detected.name}，发现 {len(chars)} 个角色：{', '.join(chars)}。接下来请调用 design_characters 设计角色。",
    }, ensure_ascii=False)


@tool
def design_characters() -> str:
    """为分析阶段识别出的角色创建详细的 Character Sheet。

    每个角色包含外貌描述（脸型、发型、体型、服装、配饰）和 SD 生图触发词。
    首次出场角色从原文提取外貌，已设计过的角色自动跳过。
    必须在 analyze_text 之后调用。
    """
    data = _ctx.data
    if not data.analysis:
        return json.dumps({"error": "请先调用 analyze_text 分析文本"})

    existing_names = {c.name for c in data.characters}
    new_chars = [p for p in data.analysis.characters_preview if p["name"] not in existing_names]

    if not new_chars:
        return json.dumps({"status": "ok", "message": "所有角色已设计，跳过。", "characters": [c.name for c in data.characters]})

    # 提取外貌相关原文
    relevant_lines = []
    for line in data.source_text.split("\n"):
        for char in new_chars:
            if char["name"] in line:
                relevant_lines.append(line.strip())
                break
    text_context = "\n".join(relevant_lines[:30])

    system_prompt = """你是专业的漫画角色设计师。为每个角色创建详细的 Character Sheet。

返回 JSON 数组：
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
    "sd_trigger_words": "英文触发词。格式: 'name, gender, hair description, clothing, distinctive feature, art style neutral'",
    "personality_notes": "性格特征对表情/姿态的影响"
  }
]

重要：sd_trigger_words 必须足够详细以确保每次生图角色外貌一致。"""

    # 从 RAG 中检索每个角色的全书上下文
    rag_context = ""
    if _ctx.rag and _ctx.rag.is_indexed:
        for char in new_chars:
            char_ctx = build_character_context(_ctx.rag, char["name"], max_chunks=3)
            if char_ctx:
                rag_context += char_ctx + "\n\n"

    user_prompt = (
        f"## 人物列表\n" + "\n".join(f"- {c['name']} ({c['role']})" for c in new_chars) +
        f"\n\n## 原文片段（含外貌描写）\n{text_context}\n\n"
        + (f"## 全书角色上下文（RAG）\n{rag_context}\n\n" if rag_context else "")
        + f"## 风格\n{data.style_profile.name if data.style_profile else 'auto'}\n\n"
        f"请为每个角色生成 Character Sheet（JSON 数组）。"
    )

    result = _llm_chat_json(system_prompt, user_prompt)

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

    names = [c.name for c in data.characters]

    # 同步到全书角色库
    if _ctx.novel:
        _ctx.novel.add_characters(data.characters)

    return json.dumps({
        "status": "ok",
        "characters": names,
        "message": f"角色设计完成。共 {len(data.characters)} 个角色：{', '.join(names)}。接下来请调用 extract_scenes 拆分场景。",
    }, ensure_ascii=False)


@tool
def extract_scenes() -> str:
    """将小说文本拆分为 3-8 个关键叙事场景。

    按地点变换、时间跳跃、情绪转折切分场景。
    每个场景包含标题、摘要、出场角色、情绪变化和关键台词。
    必须在 design_characters 之后调用。
    """
    data = _ctx.data
    if not data.characters:
        return json.dumps({"error": "请先调用 design_characters 设计角色"})

    char_names = [c.name for c in data.characters]
    style_name = data.style_profile.name if data.style_profile else "auto"

    system_prompt = """你是专业的漫画改编编剧，将小说文本拆分为适合漫画表现的场景。

拆分规则：
- 按地点变换切分（从街上到屋内 = 新场景）
- 按时间跳跃切分（"三天后" = 新场景）
- 按情绪转折切分（从平静到冲突爆发）
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

    user_prompt = (
        f"## 原文\n{data.source_text}\n\n"
        f"## 已识别角色\n{', '.join(char_names)}\n\n"
        f"## 风格\n{style_name}\n\n"
        f"请拆分为关键场景（3-8个）。"
    )

    result = _llm_chat_json(system_prompt, user_prompt)

    data.scenes = []
    for sd in result:
        scene = Scene(
            id=sd.get("id", len(data.scenes) + 1),
            title=sd.get("title", ""),
            summary=sd.get("summary", ""),
            characters_in_scene=sd.get("characters_in_scene", []),
            emotion_arc=sd.get("emotion_arc", ""),
            key_dialogue=sd.get("key_dialogue", ""),
        )
        data.scenes.append(scene)

    scene_list = [f"场景{s.id}: {s.title}" for s in data.scenes]
    return json.dumps({
        "status": "ok",
        "scene_count": len(data.scenes),
        "scenes": scene_list,
        "message": f"场景拆分完成，共 {len(data.scenes)} 个场景。接下来请逐个调用 storyboard_scene(scene_id) 为每个场景生成分镜。",
    }, ensure_ascii=False)


@tool
def storyboard_scene(scene_id: int) -> str:
    """为指定场景生成漫画分镜脚本。

    每个场景生成 3-6 个格子，每格包含：
    - 中文画面描述（含前景/中景/背景构图）
    - 角色动作和表情
    - 台词
    - 镜头角度（特写/近景/中景/远景/俯视/仰视/POV）
    - 情绪氛围
    - 英文 SD 生图 prompt（自动注入风格基座 + 角色触发词 + 画幅比例）

    Args:
        scene_id: 场景的 id 编号（从 1 开始）
    """
    data = _ctx.data
    scene = next((s for s in data.scenes if s.id == scene_id), None)
    if not scene:
        return json.dumps({"error": f"场景 {scene_id} 不存在。可用场景：{[s.id for s in data.scenes]}"})

    char_info = "\n".join(
        f"- {c.name} [{c.role}]: {c.appearance.distinctive_features} | trigger: {c.sd_trigger_words}"
        for c in data.characters
    )

    # 找到场景相关原文
    scene_chars = scene.characters_in_scene
    relevant_lines = []
    for line in data.source_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if any(ch in line for ch in scene_chars):
            relevant_lines.append(line)
    scene_text = "\n".join(relevant_lines[:20])

    system_prompt = """你是专业的漫画分镜师，精通日本漫画和韩式条漫的分镜设计。

为给定的场景生成 3-6 个格子的分镜脚本。每个格子返回：
{
  "panel_number": 1,
  "visual_description": "中文画面描述，必须有构图信息（前景/中景/背景）",
  "character_action": "角色的动作和表情",
  "dialogue": "台词（无则为空字符串）",
  "camera_angle": "特写/近景/中景/远景/俯视/仰视/主观POV/鸟瞰",
  "mood": "情绪氛围",
  "sd_prompt": "英文 SD prompt，含画风关键词 + 场景描述 + 构图提示",
  "character_refs": ["角色名列表"]
}

要求：
1. 每场景 3-6 格
2. 画面描述必须有构图感
3. 关键对话不能遗漏
4. 相邻格之间景别/视角要有变化"""

    # 从 RAG 检索场景相关上下文
    rag_context = ""
    if _ctx.rag and _ctx.rag.is_indexed:
        # 用场景标题 + 关键台词作为查询
        rag_query = f"{scene.title} {scene.summary} {scene.key_dialogue}"
        rag_context = build_context_from_search(
            _ctx.rag, rag_query, _ctx.openai_client, top_k=3, max_chars=1500,
        )

    user_prompt = (
        f"## 场景信息\n- 标题: {scene.title}\n- 摘要: {scene.summary}\n"
        f"- 情绪: {scene.emotion_arc}\n- 关键台词: {scene.key_dialogue}\n\n"
        f"## 场景原文\n{scene_text}\n\n"
        + (f"## 全书相关上下文（RAG）\n{rag_context}\n\n" if rag_context else "")
        + f"## 角色信息\n{char_info}\n\n"
        f"## 风格\n{data.style_profile.name if data.style_profile else 'auto'}\n\n"
        f"请生成 3-6 格分镜脚本（JSON 数组）。"
    )

    result = _llm_chat_json(system_prompt, user_prompt)

    def _build_prompt(panel_dict: dict) -> str:
        parts = []
        if data.style_profile:
            parts.append(data.style_profile.sd_base_prompt)
        refs = panel_dict.get("character_refs", [])
        for ref_name in refs:
            for c in data.characters:
                if c.name == ref_name and c.sd_trigger_words:
                    parts.append(c.sd_trigger_words)
        if panel_dict.get("sd_prompt"):
            parts.append(panel_dict["sd_prompt"])
        if data.style_profile:
            parts.append(f"aspect ratio {data.style_profile.aspect_ratio}")
        return ", ".join(parts)

    scene.panels = []
    for pd in result:
        panel = Panel(
            panel_number=pd.get("panel_number", len(scene.panels) + 1),
            visual_description=pd.get("visual_description", ""),
            character_action=pd.get("character_action", ""),
            dialogue=pd.get("dialogue", ""),
            camera_angle=pd.get("camera_angle", ""),
            mood=pd.get("mood", ""),
            sd_prompt=_build_prompt(pd),
            character_refs=pd.get("character_refs", scene.characters_in_scene),
        )
        scene.panels.append(panel)

    return json.dumps({
        "status": "ok",
        "scene_id": scene_id,
        "panel_count": len(scene.panels),
        "panels": [f"格{p.panel_number}: [{p.camera_angle}] {p.visual_description[:50]}..." for p in scene.panels],
        "message": f"场景{scene_id} 分镜完成，{len(scene.panels)} 格。如需调整某格，请告诉我；否则继续 storyboard_scene 下一个场景。",
    }, ensure_ascii=False)


@tool
def generate_images(scene_id: int = 0) -> str:
    """为分镜格子生成漫画图片。

    根据 sd_prompt 调用云端生图 API（或生成占位图）。
    每格生成一张图，自动匹配风格对应的画幅比例。
    生成完成后图片路径保存在对应的 Panel 中。

    Args:
        scene_id: 场景 id。0 表示生成全部场景的图片。
    """
    data = _ctx.data
    img_gen = _ctx.img_gen

    ratio_map = {"9:16": (576, 1024), "4:3": (1024, 768), "16:9": (1024, 576), "1:1": (1024, 1024)}
    sp = data.style_profile
    width, height = ratio_map.get(sp.aspect_ratio, (1024, 1024)) if sp else (1024, 1024)

    images_dir = os.path.join(data.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    scenes_to_process = [s for s in data.scenes if scene_id == 0 or s.id == scene_id]
    if not scenes_to_process:
        return json.dumps({"error": f"场景 {scene_id} 不存在"})

    generated = 0
    for scene in scenes_to_process:
        for panel in scene.panels:
            ref_path = ""
            for char_name in panel.character_refs:
                for c in data.characters:
                    if c.name == char_name and c.reference_image_path:
                        ref_path = c.reference_image_path
                        break

            path = img_gen.generate(
                prompt=panel.sd_prompt,
                output_dir=images_dir,
                width=width, height=height,
                reference_image_path=ref_path,
            )
            panel.generated_image_path = path
            panel.status = "generated"
            generated += 1

    return json.dumps({
        "status": "ok",
        "generated": generated,
        "message": f"生成了 {generated} 张图片。接下来请调用 compile_comic 排版输出。",
    }, ensure_ascii=False)


@tool
def compile_comic() -> str:
    """将已生成的图片拼接为最终漫画。

    根据风格选择排版模式：
    - webtoon/gufeng: 条漫纵向拼接 + 场景标题 + 对话框 + 格编号
    - manga: 格阵排版（暂回退为条漫模式）

    必须在 generate_images 之后调用。
    """
    import os as _os
    from PIL import Image, ImageDraw, ImageFont

    data = _ctx.data

    PANEL_GAP = 20
    MARGIN = 40
    BUBBLE_PADDING = 12
    MAX_SCROLL_WIDTH = 800

    def _load_font(size: int):
        for fp in ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simsun.ttc", "C:/Windows/Fonts/arial.ttf"]:
            if _os.path.exists(fp):
                try:
                    return ImageFont.truetype(fp, size)
                except Exception:
                    continue
        return ImageFont.load_default()

    pages = []
    for scene in data.scenes:
        panel_imgs = []
        for panel in scene.panels:
            if panel.generated_image_path and _os.path.exists(panel.generated_image_path):
                panel_imgs.append((panel, Image.open(panel.generated_image_path)))

        if not panel_imgs:
            continue

        scene_width = MAX_SCROLL_WIDTH
        resized = []
        total_h = 0
        for panel, img in panel_imgs:
            ratio = scene_width / img.width
            nh = int(img.height * ratio)
            img = img.resize((scene_width, nh), Image.LANCZOS)
            resized.append((panel, img))
            total_h += nh + PANEL_GAP

        font = _load_font(18)
        font_small = _load_font(14)
        total_h += 80 * len(resized)

        canvas = Image.new("RGB", (scene_width, total_h + MARGIN * 2), color=(30, 30, 40))
        draw = ImageDraw.Draw(canvas)
        y = MARGIN

        for panel, img in resized:
            canvas.paste(img, (0, y))
            ph = img.height

            if panel == resized[0][0]:
                title_font = _load_font(22)
                draw.text((20, y + 10), f"场景: {scene.title}", fill=(255, 255, 255), font=title_font)

            if panel.dialogue:
                text = panel.dialogue
                max_tw = scene_width - MARGIN * 2 - BUBBLE_PADDING * 2 - 40
                lines = []
                cur = ""
                for ch in list(text):
                    test = cur + ch
                    if draw.textbbox((0, 0), test, font=font)[2] > max_tw:
                        lines.append(cur)
                        cur = ch
                    else:
                        cur = test
                if cur:
                    lines.append(cur)

                lh = draw.textbbox((0, 0), "啊", font=font)[3] + 4
                th = lh * len(lines)
                bh = th + BUBBLE_PADDING * 2
                bx, bw = MARGIN + 20, scene_width - MARGIN * 2 - 40

                draw.rounded_rectangle(
                    [bx, y + ph + 10, bx + bw, y + ph + 10 + bh],
                    radius=16, fill=(255, 255, 255, 230), outline=(60, 60, 60), width=2,
                )
                ty = y + ph + 10 + BUBBLE_PADDING
                for line in lines:
                    tw = draw.textbbox((0, 0), line, font=font)[2]
                    draw.text(((scene_width - tw) // 2, ty), line, fill=(20, 20, 20), font=font)
                    ty += lh
                y += ph + bh + PANEL_GAP + 10
            else:
                y += ph + PANEL_GAP

            draw.text((scene_width - 80, y - 30), f"格{panel.panel_number}", fill=(150, 150, 170), font=font_small)

        comics_dir = _os.path.join(data.output_dir, "comics")
        _os.makedirs(comics_dir, exist_ok=True)
        op = _os.path.join(comics_dir, f"scene_{scene.id:02d}.png")
        canvas.save(op, "PNG")
        pages.append(ComicPage(page_number=scene.id, image_path=op))

    data.pages = pages
    return json.dumps({
        "status": "ok",
        "page_count": len(pages),
        "files": [p.image_path for p in pages],
        "message": f"漫画排版完成！共 {len(pages)} 页，输出目录：{data.output_dir}",
    }, ensure_ascii=False)


@tool
def save_project() -> str:
    """将当前项目状态保存到 JSON 文件。可在任何阶段调用。"""
    saved = []

    # 保存全书数据 + 同步注册表
    if _ctx.novel:
        novel_path = os.path.join(_ctx.novel.output_dir, "novel.json")
        _ctx.novel.save(novel_path)
        saved.append(novel_path)

        # 同步注册表（更新章节数、风格、访问时间）
        style = _ctx.novel.style_profile.name if _ctx.novel.style_profile else ""
        try:
            register_novel(
                _ctx.novel.file_path,
                _ctx.novel.title,
                _ctx.novel.total_chapters,
                _ctx.novel.output_dir,
                style,
            )
        except Exception:
            pass  # 注册表更新失败不影响主流程

    # 保存当前章数据
    if _ctx.chapter_data:
        ch_path = os.path.join(_ctx.chapter_data.output_dir, "chapter_data.json")
        _ctx.chapter_data.save(ch_path)
        saved.append(ch_path)

    return json.dumps({
        "status": "ok",
        "saved_files": saved,
        "novel_title": _ctx.novel.title if _ctx.novel else "",
        "chapter": _ctx.chapter_data.title if _ctx.chapter_data else "",
        "stage": _ctx.chapter_data.current_stage if _ctx.chapter_data else 0,
    }, ensure_ascii=False)


# ============================================================
# Agent 构建
# ============================================================

def build_agent():
    """使用 AgentFlow AgentBuilder 构建 Novel2Comic Agent。"""
    api_key = os.getenv("AGENTFLOW_API_KEY", "")
    base_url = os.getenv("AGENTFLOW_BASE_URL", "https://api.deepseek.com/v1")
    model = os.getenv("AGENTFLOW_MODEL", "deepseek-chat")
    proxy = os.getenv("AGENTFLOW_PROXY", "")

    if not api_key:
        print("[!] AGENTFLOW_API_KEY not set.")
        print("    请设置环境变量: $env:AGENTFLOW_API_KEY='sk-your-key'")
        sys.exit(1)

    # AgentFlow 的 LLM Client（Agent 的"大脑"）
    llm = OpenAIClient(api_key=api_key, model=model, base_url=base_url, proxy=proxy or None)

    # Tool 内部用的同步 OpenAI Client
    import httpx
    import openai
    http_client = httpx.Client(proxy=proxy) if proxy else None
    _ctx.openai_client = openai.OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
    _ctx.llm_model = model

    # ImageGen Adapter
    img_api_key = os.getenv("N2C_IMG_API_KEY", "")
    img_base_url = os.getenv("N2C_IMG_BASE_URL", "")
    _ctx.img_gen = ImageGenAdapter(api_key=img_api_key, base_url=img_base_url)

    # 构建 Agent
    agent = (AgentBuilder("novel2comic")
        .with_llm(llm)
        .with_skills_dir(os.path.join(_project_root, "skills"))
        .with_skill("novel2comic")
        .with_tools(
            load_novel,
            list_novels,
            resume_novel,
            list_chapters,
            select_chapter,
            search_novel,
            get_character_info,
            analyze_text,
            design_characters,
            extract_scenes,
            storyboard_scene,
            generate_images,
            compile_comic,
            save_project,
        )
        .with_memory(MemoryProfile.standard())
        .with_thinking(ThinkingMode.REACT)
        .with_max_iterations(30)
        .build())

    return agent


# ============================================================
# 运行入口
# ============================================================

async def run_novel_agent(novel_path: str):
    """启动 Agent：加载整本小说，让用户选择章节生成。"""
    agent = build_agent()

    task = (
        f"## 任务：加载小说并准备生成漫画\n\n"
        f"小说文件路径：{novel_path}\n\n"
        f"### 执行计划\n"
        f"1. 调用 load_novel('{novel_path}') 加载小说（如果已加载过会直接用缓存）\n"
        f"2. 调用 list_chapters 查看章节列表\n"
        f"3. 告诉我章节列表，让我选择要生成第几章\n"
        f"4. 我选择后，调用 select_chapter(N) 选中该章\n"
        f"5. 然后按顺序执行：analyze_text → design_characters → extract_scenes → storyboard_scene(每个场景) → generate_images → compile_comic → save_project\n\n"
        f"每步完成后汇报结果。如果我对某个结果不满意，我会告诉你如何调整。\n"
        f"生成完一章后，我可以叫你继续生成其他章节。\n"
        f"如果我下次想继续，用 list_novels 查看已加载的小说，resume_novel(索引) 恢复。"
    )

    print(f"\n[Agent] 加载小说: {novel_path}")
    print(f"[Agent] 模式: REACT (Agent 自主决策工具调用)\n")

    result = await agent.run(task)

    print(f"\n[Agent] 处理完成")
    print(f"[Agent] 执行步骤数: {len(result.steps)}")
    for i, step in enumerate(result.steps):
        step_type = step.get("type", step.get("phase", "?"))
        if step_type == "tool_call":
            calls = step.get("calls", [])
            for c in calls:
                print(f"  步骤{i}: [TOOL] {c.get('name', '?')}")
        else:
            output_preview = step.get("output", "")[:100]
            print(f"  步骤{i}: [{step_type}] {output_preview}...")

    print(f"\n--- Agent 回复 ---")
    print(result.output)

    return result.output


async def run_single_chapter(text: str, title: str = "未命名章节"):
    """启动 Agent：单章模式（兼容旧版用法）。"""
    agent = build_agent()

    project_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "projects",
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(project_dir, exist_ok=True)

    _ctx.chapter_data = ChapterData(
        title=title,
        source_text=text,
        output_dir=project_dir,
        created_at=datetime.now().isoformat(),
    )

    task = (
        f"## 任务：将以下小说章节转化为漫画\n\n"
        f"章节标题：{title}\n\n"
        f"### 小说原文\n{text}\n\n"
        f"### 执行计划\n"
        f"请按顺序执行：\n"
        f"1. analyze_text(text=原文)\n"
        f"2. design_characters()\n"
        f"3. extract_scenes()\n"
        f"4. 对每个场景调用 storyboard_scene(scene_id=N)\n"
        f"5. generate_images(scene_id=0)\n"
        f"6. compile_comic()\n"
        f"7. save_project()\n\n"
        f"每步完成后向我汇报结果。"
    )

    print(f"\n[Agent] 开始处理: {title}")
    print(f"[Agent] 文本长度: {len(text)} 字符")
    print(f"[Agent] 模式: REACT (Agent 自主决策工具调用)\n")

    result = await agent.run(task)

    print(f"\n[Agent] 处理完成, 步骤数: {len(result.steps)}")
    for i, step in enumerate(result.steps):
        step_type = step.get("type", step.get("phase", "?"))
        if step_type == "tool_call":
            for c in step.get("calls", []):
                print(f"  步骤{i}: [TOOL] {c.get('name', '?')}")
        else:
            print(f"  步骤{i}: [{step_type}] {step.get('output', '')[:80]}...")

    print(f"\n--- Agent 回复 ---")
    print(result.output)
    return result.output


# ============================================================
# CLI 入口
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Novel2Comic Agent V2 - Agent 驱动的小说转漫画")
        print()
        print("用法:")
        print("  全书模式: python agent.py --novel 小说.txt")
        print("  单章模式: python agent.py chapter1.txt 月下初遇")
        print('  单章模式: python agent.py "小说内容文本" 第一章')
        print()
        print("示例:")
        print("  python agent.py --novel 斗破苍穹.txt")
        print("  python agent.py chapter1.txt 月下初遇")
        sys.exit(1)

    if sys.argv[1] == "--novel":
        if len(sys.argv) < 3:
            print("[!] 请指定小说文件路径: python agent.py --novel 小说.txt")
            sys.exit(1)
        novel_path = sys.argv[2]
        asyncio.run(run_novel_agent(novel_path))
    else:
        input_text = sys.argv[1]
        chapter_title = sys.argv[2] if len(sys.argv) > 2 else "未命名章节"

        if os.path.isfile(input_text):
            with open(input_text, "r", encoding="utf-8") as f:
                input_text = f.read()
            if len(sys.argv) <= 2:
                chapter_title = os.path.splitext(os.path.basename(sys.argv[1]))[0]

        asyncio.run(run_single_chapter(input_text, chapter_title))
