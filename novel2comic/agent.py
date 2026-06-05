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
    Scene, Panel, ComicPage, StyleProfile,
)
from src.styles import detect_style, BUILTIN_STYLES
from src.img_adapter import ImageGenAdapter

# ============================================================
# 共享上下文（Tool 通过此访问 LLM / ImageGen / Data）
# ============================================================

class AgentContext:
    """Tool 共享状态——在 Agent 启动前注入。"""
    def __init__(self):
        self.data: ChapterData | None = None
        self.openai_client = None   # openai.OpenAI 同步客户端（供 Tool 内 LLM 调用）
        self.llm_model: str = ""
        self.img_gen: ImageGenAdapter | None = None

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
# 6 个 Pipeline Tool
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

    user_prompt = (
        f"## 人物列表\n" + "\n".join(f"- {c['name']} ({c['role']})" for c in new_chars) +
        f"\n\n## 原文片段（含外貌描写）\n{text_context}\n\n"
        f"## 风格\n{data.style_profile.name if data.style_profile else 'auto'}\n\n"
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

    user_prompt = (
        f"## 场景信息\n- 标题: {scene.title}\n- 摘要: {scene.summary}\n"
        f"- 情绪: {scene.emotion_arc}\n- 关键台词: {scene.key_dialogue}\n\n"
        f"## 场景原文\n{scene_text}\n\n"
        f"## 角色信息\n{char_info}\n\n"
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
    data = _ctx.data
    save_path = os.path.join(data.output_dir, "chapter_data.json")
    data.save(save_path)
    return json.dumps({
        "status": "ok",
        "path": save_path,
        "stage": data.current_stage,
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

async def run_agent(text: str, title: str = "未命名章节"):
    """启动 Agent 驱动的漫画生成流程。"""
    agent = build_agent()

    # 初始化数据总线
    project_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "projects",
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(project_dir, exist_ok=True)

    _ctx.data = ChapterData(
        title=title,
        source_text=text,
        output_dir=project_dir,
        created_at=datetime.now().isoformat(),
    )

    # 引导指令
    task = (
        f"## 任务：将以下小说章节转化为漫画\n\n"
        f"章节标题：{title}\n\n"
        f"### 小说原文\n{text}\n\n"
        f"### 执行计划\n"
        f"请按顺序执行以下步骤（每步调用对应的工具）：\n"
        f"1. 调用 analyze_text 分析文本（传入原文）\n"
        f"2. 调用 design_characters 设计角色\n"
        f"3. 调用 extract_scenes 拆分场景\n"
        f"4. 对每个场景调用 storyboard_scene(scene_id=场景id) 生成分镜\n"
        f"5. 调用 generate_images(scene_id=0) 生成全部图片\n"
        f"6. 调用 compile_comic 排版输出\n"
        f"7. 调用 save_project 保存项目\n\n"
        f"每步完成后向我汇报结果。如果我对某个结果不满意，我会告诉你如何调整。"
    )

    print(f"\n[Agent] 开始处理: {title}")
    print(f"[Agent] 文本长度: {len(text)} 字符")
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


# ============================================================
# CLI 入口
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Novel2Comic Agent V2 - Agent 驱动的小说转漫画")
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

    asyncio.run(run_agent(input_text, chapter_title))
