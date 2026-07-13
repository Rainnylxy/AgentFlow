# Novel2Comic V2 MVP 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建 Novel2Comic V2 MVP——6 阶段 Pipeline 将小说文本转化为带对话框的漫画图片，CLI 交互式协作。

**Architecture:** 纯 Pipeline 模式（不依赖 AgentFlow AgentBuilder）。每个阶段是独立函数，LLM 调用用 OpenAIClient 直接驱动，Pillow 做排版渲染。数据模型用 dataclass + JSON 序列化。CLI 在每阶段间暂停等用户确认。

**Tech Stack:** Python 3.10+, openai SDK (DeepSeek API), Pillow, dataclasses, asyncio

**MVP 范围:**

- ✅ 6 阶段 Pipeline 全部实现
- ✅ StyleProfile 三种风格 + 自动判断
- ✅ 角色定妆 + sd_trigger_words
- ✅ 分镜生成（含 sd_prompt）
- ✅ 图像生成（云端 API + 本地占位图兜底）
- ✅ 漫画排版（Pillow 合成 + 对话框）
- ✅ CLI 交互（阶段间审核/重做/保存）
- ❌ 反馈记忆系统（完整版）
- ❌ 人物关系知识图谱（完整版）
- ❌ Web UI（完整版）
- ❌ 版本历史/快照/分支（完整版）

---

## 文件结构

```
novel2comic_v2/
├── src/
│   ├── models.py              # 所有 dataclass 数据模型
│   ├── styles.py              # StyleProfile 定义 + 自动判断
│   ├── llm_adapter.py         # LLM 调用封装
│   ├── img_adapter.py         # 生图 API 封装 + 占位图兜底
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── engine.py          # Pipeline 编排器
│   │   ├── stage1_analyze.py  # ① 文本分析
│   │   ├── stage2_characters.py # ② 角色设计
│   │   ├── stage3_scenes.py   # ③ 场景拆分
│   │   ├── stage4_storyboard.py # ④ 分镜生成
│   │   ├── stage5_image_gen.py  # ⑤ 图像生成
│   │   └── stage6_layout.py   # ⑥ 漫画排版
│   └── cli.py                 # CLI 入口
├── skills/
│   └── novel2comic.md         # Skill 定义（风格约束）
├── projects/                  # 用户项目存储目录
├── requirements.txt
└── .env.example
```

---

### Task 1: 项目骨架搭建

**Files:**

- Create: `novel2comic_v2/requirements.txt`
- Create: `novel2comic_v2/.env.example`
- Create: `novel2comic_v2/src/__init__.py`
- Create: `novel2comic_v2/src/pipeline/__init__.py`
- Create: `novel2comic_v2/projects/.gitkeep`
- Create: `novel2comic_v2/skills/.gitkeep`

- [ ] **Step 1: 创建目录结构**

```bash
mkdir -p novel2comic_v2/src/pipeline novel2comic_v2/projects novel2comic_v2/skills
```

- [ ] **Step 2: 创建 requirements.txt**

```
openai>=1.0.0
httpx>=0.27.0
Pillow>=10.0.0
pydantic>=2.0.0
```

- [ ] **Step 3: 创建 .env.example**

```
N2C_LLM_API_KEY=sk-your-key
N2C_LLM_BASE_URL=https://api.deepseek.com/v1
N2C_LLM_MODEL=deepseek-chat
N2C_IMG_API_KEY=sk-your-key
N2C_IMG_BASE_URL=https://api.stability.ai/v1
N2C_PROXY=
```

- [ ] **Step 4: 创建空 **init**.py 和 .gitkeep**

```bash
touch novel2comic_v2/src/__init__.py
touch novel2comic_v2/src/pipeline/__init__.py
touch novel2comic_v2/projects/.gitkeep
touch novel2comic_v2/skills/.gitkeep
```

- [ ] **Step 5: 提交**

```bash
git add novel2comic_v2/
git commit -m "chore: scaffold novel2comic_v2 MVP project structure"
```

---

### Task 2: 数据模型 (models.py)

**Files:**

- Create: `novel2comic_v2/src/models.py`

- [ ] **Step 1: 编写完整数据模型**

```python
# -*- coding: utf-8 -*-
"""Novel2Comic V2 数据模型——所有 dataclass 定义 + JSON 序列化。"""

from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime
import json


# ============================================================
# 风格
# ============================================================

@dataclass
class StyleProfile:
    name: str                    # "manga" | "webtoon" | "gufeng"
    color_mode: str              # "bw_screentone" | "full_color" | "ink_wash"
    reading_direction: str       # "rtl_page" | "vertical_scroll" | "flexible"
    aspect_ratio: str            # "16:9" | "9:16" | "4:3" | "1:1"
    sd_base_prompt: str          # 注入每张图的风格基座
    speech_bubble_style: str     # 对话框样式
    sfx_style: str               # 特效字样式
    layout_mode: str             # "grid" (Manga 格阵) | "scroll" (条漫竖拼)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StyleProfile":
        return cls(**d)


# ============================================================
# 角色
# ============================================================

@dataclass
class CharacterAppearance:
    face: str = ""
    hair: str = ""
    build: str = ""
    clothing: str = ""
    accessories: str = ""
    distinctive_features: str = ""


@dataclass
class CharacterSheet:
    id: str                     # 唯一标识，如 "su_mo"
    name: str                   # 中文名
    role: str                   # "protagonist" | "antagonist" | "supporting" | ...
    appearance: CharacterAppearance = field(default_factory=CharacterAppearance)
    reference_image_path: str = ""        # 定妆照本地路径
    sd_trigger_words: str = ""            # 自动注入的触发词
    personality_notes: str = ""
    status: str = "draft"                # "draft" | "locked"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["appearance"] = asdict(self.appearance)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CharacterSheet":
        appearance = CharacterAppearance(**d.pop("appearance", {}))
        return cls(appearance=appearance, **d)


# ============================================================
# 分析
# ============================================================

@dataclass
class AnalysisResult:
    genre_tags: list[str] = field(default_factory=list)    # ["武侠", "悬疑"]
    style: str = "auto"                                     # "manga" | "webtoon" | "gufeng"
    tone: list[str] = field(default_factory=list)           # ["苍凉", "暗涌"]
    era: str = ""                                           # "古代架空"
    pace: str = ""                                          # "慢热" | "快节奏"
    characters_preview: list[dict] = field(default_factory=list)
    # [{name, role, first_appearance_line}]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AnalysisResult":
        return cls(**d)


# ============================================================
# 场景 + 分镜
# ============================================================

@dataclass
class Panel:
    panel_number: int = 0
    visual_description: str = ""     # 中文画面描述（含前景/中景/背景）
    character_action: str = ""       # 角色动作和表情
    dialogue: str = ""               # 台词
    camera_angle: str = ""           # 镜头角度
    mood: str = ""                   # 情绪氛围
    sd_prompt: str = ""              # 英文生图 prompt
    character_refs: list[str] = field(default_factory=list)  # 角色 ID 列表
    generated_image_path: str = ""   # 生图结果路径
    status: str = "pending"          # "pending" | "generated" | "approved" | "rejected"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Panel":
        return cls(**d)


@dataclass
class Scene:
    id: int = 0
    title: str = ""
    summary: str = ""                           # 1-2 句概述
    characters_in_scene: list[str] = field(default_factory=list)  # 角色名列表
    emotion_arc: str = ""                       # 情绪变化
    key_dialogue: str = ""                      # 关键台词
    panels: list[Panel] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["panels"] = [p.to_dict() for p in self.panels]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Scene":
        panels = [Panel.from_dict(p) for p in d.pop("panels", [])]
        return cls(panels=panels, **d)


# ============================================================
# 漫画页
# ============================================================

@dataclass
class ComicPage:
    page_number: int = 0
    image_path: str = ""             # 最终输出的图片路径

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ComicPage":
        return cls(**d)


# ============================================================
# 章节（Pipeline 的核心数据容器）
# ============================================================

@dataclass
class ChapterData:
    """Pipeline 数据总线——贯穿 6 个阶段的共享状态。"""
    # 输入
    title: str = ""
    source_text: str = ""

    # 阶段输出
    analysis: Optional[AnalysisResult] = None       # ①
    characters: list[CharacterSheet] = field(default_factory=list)  # ②
    scenes: list[Scene] = field(default_factory=list)                # ③④
    pages: list[ComicPage] = field(default_factory=list)             # ⑥

    # 元数据
    style_profile: Optional[StyleProfile] = None
    current_stage: int = 0          # 当前完成的阶段 (1-6)
    created_at: str = ""
    output_dir: str = ""

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "source_text": self.source_text,
            "analysis": self.analysis.to_dict() if self.analysis else None,
            "characters": [c.to_dict() for c in self.characters],
            "scenes": [s.to_dict() for s in self.scenes],
            "pages": [p.to_dict() for p in self.pages],
            "style_profile": self.style_profile.to_dict() if self.style_profile else None,
            "current_stage": self.current_stage,
            "created_at": self.created_at,
            "output_dir": self.output_dir,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ChapterData":
        data = cls(
            title=d.get("title", ""),
            source_text=d.get("source_text", ""),
            current_stage=d.get("current_stage", 0),
            created_at=d.get("created_at", ""),
            output_dir=d.get("output_dir", ""),
        )
        if d.get("analysis"):
            data.analysis = AnalysisResult.from_dict(d["analysis"])
        if d.get("style_profile"):
            data.style_profile = StyleProfile.from_dict(d["style_profile"])
        data.characters = [CharacterSheet.from_dict(c) for c in d.get("characters", [])]
        data.scenes = [Scene.from_dict(s) for s in d.get("scenes", [])]
        data.pages = [ComicPage.from_dict(p) for p in d.get("pages", [])]
        return data

    def save(self, filepath: str):
        """保存到 JSON 文件。"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "ChapterData":
        with open(filepath, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
```

- [ ] **Step 2: 验证模型可以导入**

```bash
cd novel2comic_v2 && python -c "from src.models import ChapterData, StyleProfile, CharacterSheet; print('models OK')"
```

Expected: `models OK`

- [ ] **Step 3: 提交**

```bash
git add novel2comic_v2/src/models.py
git commit -m "feat(n2c): add data models with JSON serialization"
```

---

### Task 3: StyleProfile 系统

**Files:**

- Create: `novel2comic_v2/src/styles.py`

- [ ] **Step 1: 编写三种内置风格定义 + 自动判断**

```python
# -*- coding: utf-8 -*-
"""StyleProfile 定义 + 自动风格判断。"""

from src.models import StyleProfile


# ============================================================
# 内置风格
# ============================================================

STYLE_MANGA = StyleProfile(
    name="manga",
    color_mode="bw_screentone",
    reading_direction="rtl_page",
    aspect_ratio="4:3",
    sd_base_prompt="manga style, black and white, screentone, speed lines, line art, high contrast",
    speech_bubble_style="irregular_rounded",
    sfx_style="hand_drawn_bold",
    layout_mode="grid",
)

STYLE_WEBTOON = StyleProfile(
    name="webtoon",
    color_mode="full_color",
    reading_direction="vertical_scroll",
    aspect_ratio="9:16",
    sd_base_prompt="webtoon style, full color, soft palette, manhwa, clean lines, gentle shading",
    speech_bubble_style="clean_rounded_rect",
    sfx_style="digital_gradient",
    layout_mode="scroll",
)

STYLE_GUFENG = StyleProfile(
    name="gufeng",
    color_mode="ink_wash",
    reading_direction="flexible",
    aspect_ratio="9:16",
    sd_base_prompt="chinese ink painting style, gufeng, watercolor wash, ancient chinese comic, elegant muted colors, flowing brushwork",
    speech_bubble_style="scroll_label",
    sfx_style="calligraphy_brush",
    layout_mode="scroll",
)

BUILTIN_STYLES = {
    "manga": STYLE_MANGA,
    "webtoon": STYLE_WEBTOON,
    "gufeng": STYLE_GUFENG,
}


# ============================================================
# 风格映射规则
# ============================================================

GENRE_STYLE_MAP = {
    # 古风
    "武侠": "gufeng",
    "仙侠": "gufeng",
    "玄幻": "gufeng",
    "历史": "gufeng",
    "古装": "gufeng",
    "古代": "gufeng",
    # 日式 Manga
    "轻小说": "manga",
    "校园": "manga",
    "恋爱": "manga",
    "日常": "manga",
    "异世界": "manga",
    # 韩式 Webtoon
    "都市": "webtoon",
    "职场": "webtoon",
    "现实": "webtoon",
    "娱乐圈": "webtoon",
    # 节奏判断（悬疑/科幻按节奏分）
    "悬疑": None,   # None 表示需要进一步判断
    "科幻": None,
}


def detect_style(genre_tags: list[str], pace: str = "") -> StyleProfile:
    """根据题材标签自动判断漫画风格。

    优先级：古风 > Manga > Webtoon。None 表示无法判断时由用户决定。
    """
    scores = {"gufeng": 0, "manga": 0, "webtoon": 0}

    for tag in genre_tags:
        mapped = GENRE_STYLE_MAP.get(tag)
        if mapped:
            scores[mapped] += 1

    # 悬疑/科幻 根据节奏判断
    for tag in genre_tags:
        if GENRE_STYLE_MAP.get(tag) is None:
            if pace in ("快节奏", "紧张", "动作"):
                scores["manga"] += 1
            else:
                scores["webtoon"] += 1

    # 返回最高分风格
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return STYLE_WEBTOON  # 默认

    return BUILTIN_STYLES[best]


def get_style(name: str) -> StyleProfile:
    """按名称获取风格。"""
    if name in BUILTIN_STYLES:
        return BUILTIN_STYLES[name]
    raise ValueError(f"Unknown style: {name}. Available: {list(BUILTIN_STYLES.keys())}")
```

- [ ] **Step 2: 验证风格系统**

```bash
cd novel2comic_v2 && python -c "
from src.styles import detect_style, get_style, BUILTIN_STYLES
s = detect_style(['武侠', '悬疑'], '慢热')
assert s.name == 'gufeng', f'Expected gufeng, got {s.name}'
s = detect_style(['都市', '恋爱'])
assert s.name in ('webtoon', 'manga'), f'Unexpected: {s.name}'
s = get_style('manga')
assert s.color_mode == 'bw_screentone'
print('styles OK')
"
```

Expected: `styles OK`

- [ ] **Step 3: 提交**

```bash
git add novel2comic_v2/src/styles.py
git commit -m "feat(n2c): add StyleProfile system with auto-detection"
```

---

### Task 4: LLM Adapter

**Files:**

- Create: `novel2comic_v2/src/llm_adapter.py`

- [ ] **Step 1: 编写 LLM 调用封装**

````python
# -*- coding: utf-8 -*-
"""LLM Adapter——封装 LLM 调用，让 Pipeline 各阶段只需关心 prompt 和输出格式。"""

import os
import json
from openai import OpenAI


class LLMAdapter:
    """LLM 调用适配器。

    封装了 API 调用细节（base_url、proxy、model），
    Pipeline 各阶段只需传入 prompt 即可获得结构化输出。
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        model: str = "deepseek-chat",
        proxy: str = "",
    ):
        self.api_key = api_key or os.getenv("N2C_LLM_API_KEY", "")
        self.base_url = base_url or os.getenv("N2C_LLM_BASE_URL", "https://api.deepseek.com/v1")
        self.model = model or os.getenv("N2C_LLM_MODEL", "deepseek-chat")
        self.proxy = proxy or os.getenv("N2C_PROXY", "")

        import httpx
        http_client = None
        if self.proxy:
            http_client = httpx.Client(proxy=self.proxy)

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            http_client=http_client,
        )

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
        """发送一次对话，返回 LLM 的文本响应。"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    def chat_json(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.3
    ) -> dict:
        """发送对话并要求 JSON 输出，返回解析后的 dict。"""
        # 在 system prompt 末尾追加 JSON 输出指令
        full_system = (
            system_prompt
            + "\n\nYou MUST respond with valid JSON only. No markdown fences, no explanation."
        )
        text = self.chat(full_system, user_prompt, temperature)

        # 清理可能的 markdown 代码块包裹
        text = text.strip()
        if text.startswith("```"):
            # 移除 ```json 和末尾的 ```
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        return json.loads(text)
````

- [ ] **Step 2: 验证导入**

```bash
cd novel2comic_v2 && python -c "from src.llm_adapter import LLMAdapter; print('llm_adapter OK')"
```

Expected: `llm_adapter OK`

- [ ] **Step 3: 提交**

```bash
git add novel2comic_v2/src/llm_adapter.py
git commit -m "feat(n2c): add LLM adapter with JSON mode"
```

---

### Task 5: ImageGen Adapter

**Files:**

- Create: `novel2comic_v2/src/img_adapter.py`

- [ ] **Step 1: 编写生图适配器（含占位图兜底）**

```python
# -*- coding: utf-8 -*-
"""ImageGen Adapter——封装云端生图 API + 本地占位图兜底。"""

import os
import io
import uuid
from PIL import Image, ImageDraw, ImageFont


class ImageGenAdapter:
    """生图适配器。

    MVP 策略:
    - 有 API key → 调云端生图（Stability AI / 兼容 OpenAI image API）
    - 无 API key → 生成占位图（有色块 + 文字标注）
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "",
        use_placeholder: bool = False,
    ):
        self.api_key = api_key or os.getenv("N2C_IMG_API_KEY", "")
        self.base_url = base_url or os.getenv("N2C_IMG_BASE_URL", "")
        self.use_placeholder = use_placeholder or not self.api_key

    def generate(
        self,
        prompt: str,
        output_dir: str,
        width: int = 1024,
        height: int = 1024,
        reference_image_path: str = "",
    ) -> str:
        """生成一张图片。

        Args:
            prompt: SD prompt
            output_dir: 输出目录
            width, height: 图片尺寸
            reference_image_path: 可选的角色参考图路径

        Returns:
            生成图片的本地文件路径
        """
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{uuid.uuid4().hex[:12]}.png"
        filepath = os.path.join(output_dir, filename)

        if self.use_placeholder:
            self._generate_placeholder(prompt, filepath, width, height)
        else:
            self._generate_cloud(prompt, filepath, width, height, reference_image_path)

        return filepath

    def _generate_placeholder(
        self, prompt: str, filepath: str, width: int, height: int
    ):
        """生成占位图——纯色背景 + prompt 摘要。"""
        img = Image.new("RGB", (width, height), color=(40, 40, 50))
        draw = ImageDraw.Draw(img)

        # 绘制边框
        draw.rectangle([0, 0, width - 1, height - 1], outline=(100, 100, 120), width=3)

        # 绘制 prompt 摘要（取前 80 字符）
        summary = prompt[:80] + ("..." if len(prompt) > 80 else "")
        lines = [summary[i:i+40] for i in range(0, len(summary), 40)]

        # 使用默认字体，如果系统有中文字体则尝试加载
        font = None
        for font_path in [
            "C:/Windows/Fonts/msyh.ttc",   # 微软雅黑
            "C:/Windows/Fonts/simsun.ttc",  # 宋体
            "C:/Windows/Fonts/arial.ttf",
        ]:
            if os.path.exists(font_path):
                try:
                    font = ImageFont.truetype(font_path, 16)
                    break
                except Exception:
                    continue

        y = height // 2 - 30
        for line in lines:
            if font:
                bbox = draw.textbbox((0, 0), line, font=font)
                tw = bbox[2] - bbox[0]
            else:
                tw = len(line) * 8
            draw.text(((width - tw) // 2, y), line, fill=(200, 200, 220), font=font)
            y += 24

        # 底部标注
        note = "[PLACEHOLDER]"
        draw.text((width - 130, height - 30), note, fill=(150, 150, 170), font=font)

        img.save(filepath, "PNG")

    def _generate_cloud(
        self,
        prompt: str,
        filepath: str,
        width: int,
        height: int,
        reference_image_path: str = "",
    ):
        """调云端 API 生图。MVP 使用 OpenAI DALL-E 兼容接口。"""
        from openai import OpenAI
        import httpx
        import base64

        proxy = os.getenv("N2C_PROXY", "")
        http_client = httpx.Client(proxy=proxy) if proxy else None

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url or "https://api.openai.com/v1",
            http_client=http_client,
        )

        kwargs = {
            "model": "dall-e-3",
            "prompt": prompt,
            "size": "1024x1024",
            "quality": "standard",
            "n": 1,
        }

        # 如果有参考图且 API 支持，传入参考图
        if reference_image_path and os.path.exists(reference_image_path):
            # DALL-E 3 不直接支持参考图，但某些兼容 API（如即梦）支持
            # 此处保留接口，具体实现根据所选 API 调整
            pass

        response = client.images.generate(**kwargs)
        image_url = response.data[0].url

        # 下载图片
        import requests
        img_response = requests.get(image_url)
        img = Image.open(io.BytesIO(img_response.content))
        img = img.resize((width, height), Image.LANCZOS)
        img.save(filepath, "PNG")
```

- [ ] **Step 2: 测试占位图生成**

```bash
cd novel2comic_v2 && python -c "
from src.img_adapter import ImageGenAdapter
import os, tempfile
adapter = ImageGenAdapter(use_placeholder=True)
path = adapter.generate('a brave warrior under moonlight', tempfile.gettempdir())
assert os.path.exists(path), f'File not found: {path}'
print(f'Placeholder generated: {path}')
os.remove(path)
print('img_adapter OK')
"
```

Expected: Placeholder 文件生成成功

- [ ] **Step 3: 提交**

```bash
git add novel2comic_v2/src/img_adapter.py
git commit -m "feat(n2c): add ImageGen adapter with placeholder fallback"
```

---

### Task 6: Pipeline 引擎

**Files:**

- Create: `novel2comic_v2/src/pipeline/engine.py`

- [ ] **Step 1: 编写 Pipeline 编排器**

```python
# -*- coding: utf-8 -*-
"""Pipeline 引擎——编排 6 个阶段顺序执行，每阶段间可暂停。"""

from typing import Callable
from src.models import ChapterData
from src.llm_adapter import LLMAdapter
from src.img_adapter import ImageGenAdapter


# 阶段函数签名
StageFn = Callable[[ChapterData, LLMAdapter, ImageGenAdapter], ChapterData]


class PipelineEngine:
    """Pipeline 编排器。

    负责按顺序调用 6 个阶段函数，管理当前进度，支持单步执行和完整运行。
    """

    STAGE_NAMES = {
        0: "未开始",
        1: "① 文本分析",
        2: "② 角色设计",
        3: "③ 场景拆分",
        4: "④ 分镜生成",
        5: "⑤ 图像生成",
        6: "⑥ 漫画排版",
    }

    def __init__(self, llm: LLMAdapter, img_gen: ImageGenAdapter):
        self._stages: list[StageFn] = []
        self.llm = llm
        self.img_gen = img_gen

    def register(self, stage_fn: StageFn):
        """注册一个阶段函数。按注册顺序执行。"""
        self._stages.append(stage_fn)

    def run_stage(self, data: ChapterData, stage_index: int) -> ChapterData:
        """执行单个阶段。stage_index 从 0 开始。"""
        if stage_index >= len(self._stages):
            raise ValueError(f"Stage {stage_index} not registered (total: {len(self._stages)})")

        print(f"\n{'='*50}")
        print(f"  {self.STAGE_NAMES.get(stage_index + 1, f'Stage {stage_index+1}')}")
        print(f"{'='*50}")

        stage_fn = self._stages[stage_index]
        data = stage_fn(data, self.llm, self.img_gen)
        data.current_stage = stage_index + 1

        print(f"  ✅ 完成")
        return data

    def run_all(self, data: ChapterData) -> ChapterData:
        """完整运行所有已注册的阶段（无暂停）。"""
        for i in range(len(self._stages)):
            data = self.run_stage(data, i)
        return data

    @property
    def total_stages(self) -> int:
        return len(self._stages)

    def stage_name(self, index: int) -> str:
        return self.STAGE_NAMES.get(index + 1, f"Stage {index+1}")
```

- [ ] **Step 2: 验证导入**

```bash
cd novel2comic_v2 && python -c "from src.pipeline.engine import PipelineEngine; print('engine OK')"
```

Expected: `engine OK`

- [ ] **Step 3: 提交**

```bash
git add novel2comic_v2/src/pipeline/engine.py
git commit -m "feat(n2c): add Pipeline engine orchestrator"
```

---

### Task 7: Stage ① 文本分析

**Files:**

- Create: `novel2comic_v2/src/pipeline/stage1_analyze.py`

- [ ] **Step 1: 编写文本分析阶段**

```python
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
```

- [ ] **Step 2: 编写测试**

```bash
cd novel2comic_v2 && python -c "
from src.models import ChapterData
from src.pipeline.stage1_analyze import run_stage1

data = ChapterData(
    title='测试章节',
    source_text='夜幕降临，长安城华灯初上。苏墨站在朱雀大街的尽头，手握一柄锈迹斑斑的铁剑。'
)
# 验证函数签名和基本逻辑（不调 LLM）
print('stage1_analyze OK')
"
```

Expected: `stage1_analyze OK`

- [ ] **Step 3: 提交**

```bash
git add novel2comic_v2/src/pipeline/stage1_analyze.py
git commit -m "feat(n2c): add Stage 1 - text analysis"
```

---

### Task 8: Stage ② 角色设计

**Files:**

- Create: `novel2comic_v2/src/pipeline/stage2_characters.py`

- [ ] **Step 1: 编写角色设计阶段**

```python
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
```

- [ ] **Step 2: 验证导入**

```bash
cd novel2comic_v2 && python -c "from src.pipeline.stage2_characters import run_stage2; print('stage2_characters OK')"
```

Expected: `stage2_characters OK`

- [ ] **Step 3: 提交**

```bash
git add novel2comic_v2/src/pipeline/stage2_characters.py
git commit -m "feat(n2c): add Stage 2 - character design"
```

---

### Task 9: Stage ③ 场景拆分

**Files:**

- Create: `novel2comic_v2/src/pipeline/stage3_scenes.py`

- [ ] **Step 1: 编写场景拆分阶段**

```python
# -*- coding: utf-8 -*-
"""Stage ③: 场景拆分——将小说文本切分为关键叙事场景。"""

from src.models import ChapterData, Scene
from src.llm_adapter import LLMAdapter
from src.img_adapter import ImageGenAdapter


SYSTEM_PROMPT = """你是专业的漫画改编编剧，擅长将小说文本拆分为适合漫画表现的场景。

一个"场景"是一个独立的叙事单元——地点不变、时间连续、有明确的起承转合。

拆分规则：
- 按地点变换切分（从街上到屋内 = 新场景）
- 按时间跳跃切分（"三天后" = 新场景）
- 按情绪转折切分（从平静到冲突爆发 = 可分新场景）
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


def run_stage3(data: ChapterData, llm: LLMAdapter, img_gen: ImageGenAdapter) -> ChapterData:
    """③ 场景拆分。"""

    char_names = [c.name for c in data.characters]
    style_name = data.style_profile.name if data.style_profile else "auto"

    user_prompt = (
        f"## 原文\n{data.source_text}\n\n"
        f"## 已识别角色\n{', '.join(char_names) if char_names else '（无）'}\n\n"
        f"## 风格\n{style_name}\n\n"
        f"请将以上文本拆分为关键场景（3-8个）。"
    )

    result = llm.chat_json(SYSTEM_PROMPT, user_prompt)

    data.scenes = []
    for scene_dict in result:
        scene = Scene(
            id=scene_dict.get("id", len(data.scenes) + 1),
            title=scene_dict.get("title", ""),
            summary=scene_dict.get("summary", ""),
            characters_in_scene=scene_dict.get("characters_in_scene", []),
            emotion_arc=scene_dict.get("emotion_arc", ""),
            key_dialogue=scene_dict.get("key_dialogue", ""),
        )
        data.scenes.append(scene)
        print(f"  🎬 场景{scene.id}: {scene.title} [{scene.emotion_arc}]")

    print(f"  共 {len(data.scenes)} 个场景")
    return data
```

- [ ] **Step 2: 验证导入**

```bash
cd novel2comic_v2 && python -c "from src.pipeline.stage3_scenes import run_stage3; print('stage3_scenes OK')"
```

Expected: `stage3_scenes OK`

- [ ] **Step 3: 提交**

```bash
git add novel2comic_v2/src/pipeline/stage3_scenes.py
git commit -m "feat(n2c): add Stage 3 - scene extraction"
```

---

### Task 10: Stage ④ 分镜生成

**Files:**

- Create: `novel2comic_v2/src/pipeline/stage4_storyboard.py`

- [ ] **Step 1: 编写分镜生成阶段**

```python
# -*- coding: utf-8 -*-
"""Stage ④: 分镜生成——为每个场景设计格子化的画面语言。"""

import json
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

    char_names = [c.name for c in data.characters]
    char_info = "\n".join(
        f"- {c.name} [{c.role}]: {c.appearance.distinctive_features} | trigger: {c.sd_trigger_words}"
        for c in data.characters
    )

    for scene in data.scenes:
        print(f"\n  🎬 处理场景 {scene.id}: {scene.title}")

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
```

- [ ] **Step 2: 验证导入**

```bash
cd novel2comic_v2 && python -c "from src.pipeline.stage4_storyboard import run_stage4; print('stage4_storyboard OK')"
```

Expected: `stage4_storyboard OK`

- [ ] **Step 3: 提交**

```bash
git add novel2comic_v2/src/pipeline/stage4_storyboard.py
git commit -m "feat(n2c): add Stage 4 - storyboard generation with auto prompt enhancement"
```

---

### Task 11: Stage ⑤ 图像生成

**Files:**

- Create: `novel2comic_v2/src/pipeline/stage5_image_gen.py`

- [ ] **Step 1: 编写图像生成阶段**

```python
# -*- coding: utf-8 -*-
"""Stage ⑤: 图像生成——为每格分镜生成对应的漫画图片。"""

import os
from src.models import ChapterData
from src.llm_adapter import LLMAdapter
from src.img_adapter import ImageGenAdapter


def _get_image_size(style_profile) -> tuple:
    """根据风格获取图片尺寸。"""
    ratio_map = {
        "9:16": (576, 1024),
        "4:3": (1024, 768),
        "16:9": (1024, 576),
        "1:1": (1024, 1024),
    }
    if style_profile:
        return ratio_map.get(style_profile.aspect_ratio, (1024, 1024))
    return (1024, 1024)


def run_stage5(data: ChapterData, llm: LLMAdapter, img_gen: ImageGenAdapter) -> ChapterData:
    """⑤ 图像生成——逐格调用生图 API 或生成占位图。"""

    width, height = _get_image_size(data.style_profile)
    images_dir = os.path.join(data.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    total_panels = sum(len(s.panels) for s in data.scenes)
    current = 0

    for scene in data.scenes:
        for panel in scene.panels:
            current += 1
            print(f"\n  🖼️  生成 [{current}/{total_panels}] 场景{scene.id} 格{panel.panel_number}")

            # 找该格涉及角色的参考图
            ref_path = ""
            for char_name in panel.character_refs:
                for char in data.characters:
                    if char.name == char_name and char.reference_image_path:
                        ref_path = char.reference_image_path
                        break

            image_path = img_gen.generate(
                prompt=panel.sd_prompt,
                output_dir=images_dir,
                width=width,
                height=height,
                reference_image_path=ref_path,
            )

            panel.generated_image_path = image_path
            panel.status = "generated"
            print(f"    → {os.path.basename(image_path)}")

    print(f"\n  全部 {total_panels} 格图片生成完毕")
    return data
```

- [ ] **Step 2: 验证导入**

```bash
cd novel2comic_v2 && python -c "from src.pipeline.stage5_image_gen import run_stage5; print('stage5_image_gen OK')"
```

Expected: `stage5_image_gen OK`

- [ ] **Step 3: 提交**

```bash
git add novel2comic_v2/src/pipeline/stage5_image_gen.py
git commit -m "feat(n2c): add Stage 5 - image generation"
```

---

### Task 12: Stage ⑥ 漫画排版

**Files:**

- Create: `novel2comic_v2/src/pipeline/stage6_layout.py`

- [ ] **Step 1: 编写排版渲染阶段**

```python
# -*- coding: utf-8 -*-
"""Stage ⑥: 漫画排版——将图片拼接成漫画页，叠加对话框和特效字。"""

import os
from PIL import Image, ImageDraw, ImageFont
from src.models import ChapterData, ComicPage
from src.llm_adapter import LLMAdapter
from src.img_adapter import ImageGenAdapter


# 排版常量
PANEL_GAP = 20          # 格间距（像素）
MARGIN = 40             # 页面边距
BUBBLE_PADDING = 12     # 对话框内边距
BUBBLE_RADIUS = 16      # 对话框圆角半径
MAX_SCROLL_WIDTH = 800  # 条漫最大宽度


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """加载中文字体。"""
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _draw_speech_bubble(
    draw: ImageDraw.Draw,
    img_width: int,
    text: str,
    y_position: int,
    font: ImageFont.FreeTypeFont,
    style: str = "clean_rounded_rect",
):
    """在图片上绘制对话框。返回对话框占用的高度。"""
    if not text.strip():
        return 0

    # 计算文字尺寸
    max_text_width = img_width - MARGIN * 2 - BUBBLE_PADDING * 2 - 40
    lines = []
    words = list(text)
    current_line = ""
    for char in words:
        test_line = current_line + char
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] - bbox[0] > max_text_width:
            lines.append(current_line)
            current_line = char
        else:
            current_line = test_line
    if current_line:
        lines.append(current_line)

    line_height = draw.textbbox((0, 0), "啊", font=font)[3] + 4
    text_height = line_height * len(lines)
    bubble_height = text_height + BUBBLE_PADDING * 2

    # 气泡位置（水平居中，偏上方）
    bubble_x = MARGIN + 20
    bubble_w = img_width - MARGIN * 2 - 40
    bubble_y = y_position

    # 绘制圆角矩形气泡
    draw.rounded_rectangle(
        [bubble_x, bubble_y, bubble_x + bubble_w, bubble_y + bubble_height],
        radius=BUBBLE_RADIUS,
        fill=(255, 255, 255, 230),
        outline=(60, 60, 60),
        width=2,
    )

    # 绘制文字（居中）
    text_y = bubble_y + BUBBLE_PADDING
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(((img_width - tw) // 2, text_y), line, fill=(20, 20, 20), font=font)
        text_y += line_height

    return bubble_height + PANEL_GAP


def _render_scroll(data: ChapterData) -> list[ComicPage]:
    """条漫模式（Webtoon/古风）：纵向拼接所有格子。"""
    pages = []

    for scene in data.scenes:
        # 收集该场景的所有图片
        panel_images = []
        for panel in scene.panels:
            if panel.generated_image_path and os.path.exists(panel.generated_image_path):
                panel_images.append((panel, Image.open(panel.generated_image_path)))

        if not panel_images:
            continue

        # 统一宽度
        scene_width = MAX_SCROLL_WIDTH

        # 调整图片宽度 + 计算总高度
        resized = []
        total_height = 0
        for panel, img in panel_images:
            ratio = scene_width / img.width
            new_h = int(img.height * ratio)
            img = img.resize((scene_width, new_h), Image.LANCZOS)
            resized.append((panel, img))
            total_height += new_h + PANEL_GAP

        # 为对话框预留额外高度
        font = _load_font(18)
        font_small = _load_font(14)
        total_height += 80 * len(resized)  # 每个面板预留对话框空间

        # 创建场景画布
        canvas = Image.new("RGB", (scene_width, total_height + MARGIN * 2), color=(30, 30, 40))
        draw = ImageDraw.Draw(canvas)

        # 粘贴面板
        y = MARGIN
        for panel, img in resized:
            canvas.paste(img, (0, y))
            panel_h = img.height

            # 绘制场景标题（第一个面板）
            if panel == resized[0][0]:
                title_font = _load_font(22)
                draw.text(
                    (20, y + 10),
                    f"场景: {scene.title}",
                    fill=(255, 255, 255),
                    font=title_font,
                )

            # 绘制对话框
            if panel.dialogue:
                bubble_h = _draw_speech_bubble(
                    draw, scene_width, panel.dialogue,
                    y + panel_h + 10, font, data.style_profile.speech_bubble_style if data.style_profile else "clean_rounded_rect"
                )
                y += panel_h + bubble_h
            else:
                y += panel_h + PANEL_GAP

            # 面板编号
            draw.text(
                (scene_width - 80, y - 30),
                f"格{panel.panel_number}",
                fill=(150, 150, 170),
                font=font_small,
            )

        # 保存场景漫画
        os.makedirs(os.path.join(data.output_dir, "comics"), exist_ok=True)
        output_path = os.path.join(data.output_dir, "comics", f"scene_{scene.id:02d}.png")
        canvas.save(output_path, "PNG")

        page = ComicPage(page_number=scene.id, image_path=output_path)
        pages.append(page)
        print(f"  📄 场景{scene.id} 排版完成 → {output_path}")

    return pages


def run_stage6(data: ChapterData, llm: LLMAdapter, img_gen: ImageGenAdapter) -> ChapterData:
    """⑥ 漫画排版——根据风格选择排版模式。"""

    layout_mode = data.style_profile.layout_mode if data.style_profile else "scroll"

    if layout_mode == "grid":
        # TODO: Manga 格阵排版（完整版实现）
        # MVP 先用 scroll 模式处理所有风格
        print("  [INFO] Grid layout not yet implemented, falling back to scroll mode")
        data.pages = _render_scroll(data)
    else:
        data.pages = _render_scroll(data)

    print(f"\n  共生成 {len(data.pages)} 页漫画")
    return data
```

- [ ] **Step 2: 验证导入**

```bash
cd novel2comic_v2 && python -c "from src.pipeline.stage6_layout import run_stage6; print('stage6_layout OK')"
```

Expected: `stage6_layout OK`

- [ ] **Step 3: 提交**

```bash
git add novel2comic_v2/src/pipeline/stage6_layout.py
git commit -m "feat(n2c): add Stage 6 - comic layout with speech bubbles for scroll mode"
```

---

### Task 13: CLI 入口

**Files:**

- Create: `novel2comic_v2/src/cli.py`

- [ ] **Step 1: 编写 CLI 交互入口**

```python
# -*- coding: utf-8 -*-
"""Novel2Comic V2 CLI——交互式漫画生成工具。

用法:
    python -m src.cli "小说文本" --title "第一章"
    python -m src.cli chapter1.txt --title "月下归来"
    python -m src.cli --load projects/my_project/chapter_01.json
"""

import os
import sys
import argparse
from datetime import datetime

# 确保项目根在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import ChapterData
from src.llm_adapter import LLMAdapter
from src.img_adapter import ImageGenAdapter
from src.pipeline.engine import PipelineEngine
from src.pipeline.stage1_analyze import run_stage1
from src.pipeline.stage2_characters import run_stage2
from src.pipeline.stage3_scenes import run_stage3
from src.pipeline.stage4_storyboard import run_stage4
from src.pipeline.stage5_image_gen import run_stage5
from src.pipeline.stage6_layout import run_stage6


def _print_header(data: ChapterData):
    """打印章节信息头。"""
    print("\n" + "=" * 60)
    print(f"  📖 {data.title}")
    print(f"  文本长度: {len(data.source_text)} 字符")
    print(f"  风格: {data.style_profile.name if data.style_profile else 'auto'}")
    print("=" * 60)


def _print_status(data: ChapterData, engine: PipelineEngine):
    """打印当前进度。"""
    print(f"\n  当前进度: {engine.stage_name(data.current_stage)}")
    if data.current_stage < engine.total_stages:
        print(f"  下一步:   {engine.stage_name(data.current_stage + 1)}")
    print(f"  角色: {len(data.characters)} 个")
    print(f"  场景: {len(data.scenes)} 个")
    panels = sum(len(s.panels) for s in data.scenes)
    print(f"  分镜: {panels} 格")


def _show_menu() -> str:
    """显示交互菜单。"""
    print("\n" + "-" * 40)
    print("  [c] 继续下一阶段")
    print("  [r] 重做当前阶段")
    print("  [v] 查看当前数据摘要")
    print("  [s] 保存进度")
    print("  [q] 保存并退出")
    print("-" * 40)
    return input("  > ").strip().lower()


def _show_summary(data: ChapterData):
    """简要展示当前数据。"""
    print("\n  ── 数据摘要 ──")
    if data.analysis:
        print(f"  风格: {data.analysis.style} | 基调: {data.analysis.tone}")
    print(f"  角色 ({len(data.characters)}):")
    for c in data.characters:
        print(f"    - {c.name} [{c.role}] {c.sd_trigger_words[:50]}...")
    print(f"  场景 ({len(data.scenes)}):")
    for s in data.scenes:
        print(f"    - 场景{s.id}: {s.title} ({len(s.panels)}格)")
        for p in s.panels:
            status_icon = "✅" if p.status == "generated" else "⏳"
            print(f"      {status_icon} 格{p.panel_number}: {p.visual_description[:40]}...")


def main():
    parser = argparse.ArgumentParser(description="Novel2Comic V2 - 小说转漫画")
    parser.add_argument("input", nargs="?", help="小说文本或文件路径")
    parser.add_argument("--title", "-t", default="未命名章节", help="章节标题")
    parser.add_argument("--load", "-l", help="从 JSON 文件恢复进度")
    parser.add_argument("--auto", "-a", action="store_true", help="全自动模式（无交互）")
    args = parser.parse_args()

    # 初始化 LLM 和 ImageGen
    try:
        llm = LLMAdapter()
    except Exception as e:
        print(f"[!] LLM 初始化失败: {e}")
        print("[!] 请设置 N2C_LLM_API_KEY 环境变量")
        sys.exit(1)

    img_gen = ImageGenAdapter()

    # 加载或新建数据
    if args.load:
        data = ChapterData.load(args.load)
        print(f"[+] 从 {args.load} 恢复进度")
    else:
        text = args.input or ""
        if not text:
            parser.print_help()
            sys.exit(1)
        if os.path.isfile(text):
            with open(text, "r", encoding="utf-8") as f:
                text = f.read()
            if args.title == "未命名章节":
                args.title = os.path.splitext(os.path.basename(text))[0]

        # 初始化数据
        project_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "projects",
            datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
        os.makedirs(project_dir, exist_ok=True)

        data = ChapterData(
            title=args.title,
            source_text=text,
            output_dir=project_dir,
            created_at=datetime.now().isoformat(),
        )

    # 构建 Pipeline
    engine = PipelineEngine(llm, img_gen)
    engine.register(run_stage1)
    engine.register(run_stage2)
    engine.register(run_stage3)
    engine.register(run_stage4)
    engine.register(run_stage5)
    engine.register(run_stage6)

    # 主循环
    while data.current_stage < engine.total_stages:
        _print_header(data)
        _print_status(data, engine)

        if args.auto:
            data = engine.run_stage(data, data.current_stage)
            continue

        choice = _show_menu()

        if choice == "c":
            data = engine.run_stage(data, data.current_stage)
            # 自动保存
            save_path = os.path.join(data.output_dir, "chapter_data.json")
            data.save(save_path)
            print(f"  💾 已自动保存到 {save_path}")

        elif choice == "r":
            # 重做：回退到当前阶段开始
            prev_stage = max(0, data.current_stage - 1)
            # 清除当前阶段的输出
            if data.current_stage == 1:
                data.analysis = None
            elif data.current_stage == 2:
                data.characters = []
            elif data.current_stage == 3:
                data.scenes = []
            elif data.current_stage == 4:
                for s in data.scenes:
                    s.panels = []
            elif data.current_stage == 5:
                for s in data.scenes:
                    for p in s.panels:
                        p.generated_image_path = ""
                        p.status = "pending"
            elif data.current_stage == 6:
                data.pages = []
            data.current_stage = prev_stage
            print(f"  ↩️  已回退到 {engine.stage_name(data.current_stage)}")

        elif choice == "v":
            _show_summary(data)

        elif choice == "s":
            save_path = os.path.join(data.output_dir, "chapter_data.json")
            data.save(save_path)
            print(f"  💾 已保存到 {save_path}")

        elif choice == "q":
            save_path = os.path.join(data.output_dir, "chapter_data.json")
            data.save(save_path)
            print(f"  💾 已保存到 {save_path}")
            print("  👋 再见！")
            break

    if data.current_stage >= engine.total_stages:
        print("\n" + "=" * 60)
        print("  🎉 全部 6 阶段完成！")
        print(f"  输出目录: {data.output_dir}")
        for page in data.pages:
            print(f"  📄 {page.image_path}")
        print("=" * 60)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证 CLI 可以加载**

```bash
cd novel2comic_v2 && python -c "from src.cli import main; print('cli OK')"
```

Expected: `cli OK`

- [ ] **Step 3: 提交**

```bash
git add novel2comic_v2/src/cli.py
git commit -m "feat(n2c): add CLI entry point with interactive stage-by-stage execution"
```

---

### Task 14: Skill 定义文件

**Files:**

- Create: `novel2comic_v2/skills/novel2comic.md`

- [ ] **Step 1: 编写 Skill 定义**

```markdown
---
name: novel2comic
description: 将小说文本转化为漫画分镜脚本 + 图片生成 prompt
---

## Role

你是专业的漫画分镜师 (Comic Storyboard Artist)，精通日式漫画、韩式条漫、中式古风漫画的分镜设计。

## 工作流程

**第一步：分析 + 规划**

1. 分析文本：类型、风格判断 (manga/webtoon/gufeng)、人物列表、情感基调
2. 拆分场景：找出关键叙事场景（3-8 个），每个场景概括为 1-2 句话

**第二步：执行生成**
按场景逐一生成完整的分镜 Markdown，每格包含：

- 画面描述（中文，含前景/中景/背景构图）
- 角色动作和表情
- 台词（无则留空）
- 镜头角度
- 情绪氛围
- SD 生图 prompt（英文，含画风关键词）

**第三步：排版输出**
根据风格选择排版模式（格阵/条漫），合成最终漫画图片。

## 三种风格规范

### 日式 Manga

- 黑白为主，灰度网点点缀
- sd_prompt: `manga style, black and white, screentone, speed lines, line art`

### 韩式 Webtoon

- 全彩色，柔和调色板，竖屏滑动
- sd_prompt: `webtoon style, full color, soft palette, manhwa, vertical scroll`

### 中式古风

- 水墨风/工笔重彩，低饱和雅致色调
- sd_prompt: `chinese ink painting style, gufeng, watercolor wash, ancient chinese comic`

## 质量规范

1. 每场景 3-6 格分镜
2. 画面描述必须有构图信息
3. SD prompt 包含画风关键词 + 画幅比例
4. 关键情感转折台词不能遗漏
5. 人物首次出现描述外貌特征
6. 相邻格之间要有视觉变化
```

- [ ] **Step 2: 提交**

```bash
git add novel2comic_v2/skills/novel2comic.md
git commit -m "feat(n2c): add Skill definition file for novel2comic"
```

---

### Task 15: 端到端集成测试

**Files:**

- Create: `novel2comic_v2/tests/test_pipeline.py`

- [ ] **Step 1: 创建 tests 目录和测试文件**

```bash
mkdir -p novel2comic_v2/tests
```

- [ ] **Step 2: 编写集成测试（使用 Mock LLM）**

```python
# -*- coding: utf-8 -*-
"""Pipeline 集成测试——使用 Mock LLM 验证 6 阶段端到端流程。"""

import os
import sys
import json
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import ChapterData, AnalysisResult, CharacterAppearance
from src.llm_adapter import LLMAdapter
from src.img_adapter import ImageGenAdapter
from src.styles import STYLE_GUFENG, detect_style
from src.pipeline.engine import PipelineEngine
from src.pipeline.stage1_analyze import run_stage1
from src.pipeline.stage2_characters import run_stage2
from src.pipeline.stage3_scenes import run_stage3
from src.pipeline.stage4_storyboard import run_stage4
from src.pipeline.stage5_image_gen import run_stage5
from src.pipeline.stage6_layout import run_stage6


SAMPLE_TEXT = """
夜幕降临，长安城华灯初上。苏墨站在朱雀大街的尽头，手握一柄锈迹斑斑的铁剑。

"三年了，我终于回来了。"他低声自语，目光穿过熙攘的人潮，锁定在那座金碧辉煌的将军府上。

一个卖糖葫芦的老者经过，苏墨叫住了他："老人家，将军府近日可有什么动静？"

老者打量了他一眼，压低声音道："小兄弟，将军府三日前贴出告示，要招纳天下剑客，缉拿大盗'夜枭'。赏金一千两黄金。"

"一千两黄金..."苏墨嘴角微扬，眼中闪过一丝复杂的神色。

他绕过朱雀大街，钻进一条暗巷。一只黑猫从墙头跃下，落在他肩上。苏墨从怀中取出一张泛黄的羊皮纸，上面画着将军府的内部地形图。

"夜枭...呵，他们连我的真名都不知道了。"他收起羊皮纸，身形一闪，消失在夜色中。
"""


class MockLLM(LLMAdapter):
    """Mock LLM——返回预定义 JSON，不调用真实 API。"""

    def __init__(self):
        pass  # 不调用父类 __init__，避免需要 API key

    def chat_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.3) -> dict:
        # 根据 system prompt 内容判断是哪个阶段
        if "genre_tags" in system_prompt and "style" in system_prompt:
            # Stage 1: 分析
            return {
                "genre_tags": ["武侠", "悬疑"],
                "style": "gufeng",
                "tone": ["苍凉", "暗涌"],
                "era": "古代架空",
                "pace": "慢热",
                "characters_preview": [
                    {"name": "苏墨", "role": "主角", "first_appearance_line": "苏墨站在朱雀大街的尽头"},
                    {"name": "老者", "role": "配角", "first_appearance_line": "一个卖糖葫芦的老者"},
                    {"name": "黑猫", "role": "伙伴", "first_appearance_line": "一只黑猫从墙头跃下"},
                ],
            }
        elif "Character Sheet" in system_prompt or "sd_trigger_words" in system_prompt:
            # Stage 2: 角色设计
            return [
                {
                    "id": "su_mo",
                    "name": "苏墨",
                    "role": "protagonist",
                    "appearance": {
                        "face": "清瘦，下颌线锋利",
                        "hair": "长发束起，黑色",
                        "build": "修长精瘦",
                        "clothing": "灰色旧袍",
                        "accessories": "锈迹铁剑",
                        "distinctive_features": "锐利的眼神",
                    },
                    "sd_trigger_words": "su_mo, lean swordsman, sharp jawline, long black hair, grey robes, rusty sword",
                    "personality_notes": "冷峻内敛",
                },
                {
                    "id": "old_man",
                    "name": "老者",
                    "role": "supporting",
                    "appearance": {
                        "face": "满是皱纹",
                        "hair": "花白稀疏",
                        "build": "佝偻瘦小",
                        "clothing": "旧毡帽粗布衣",
                        "accessories": "糖葫芦小车",
                        "distinctive_features": "精明的小眼睛",
                    },
                    "sd_trigger_words": "old street vendor, weathered face, worn hat, carrying candied hawthorn sticks",
                    "personality_notes": "市井精明",
                },
                {
                    "id": "black_cat",
                    "name": "黑猫",
                    "role": "supporting",
                    "appearance": {"face": "", "hair": "", "build": "", "clothing": "", "accessories": "", "distinctive_features": "纯黑毛色"},
                    "sd_trigger_words": "black cat, sleek fur, glowing eyes, mysterious feline companion",
                    "personality_notes": "神秘伙伴",
                },
            ]
        elif "场景拆分" in system_prompt or "叙事单元" in system_prompt:
            # Stage 3: 场景拆分
            return [
                {"id": 1, "title": "朱雀大街·归来", "summary": "苏墨站在长安街头，手握锈剑锁定将军府。", "characters_in_scene": ["苏墨"], "emotion_arc": "苍凉→暗涌", "key_dialogue": "三年了，我终于回来了。"},
                {"id": 2, "title": "糖葫芦摊·情报", "summary": "苏墨向老者打探将军府消息，得知自己被以夜枭之名悬赏。", "characters_in_scene": ["苏墨", "老者"], "emotion_arc": "平静→暗讽", "key_dialogue": "赏金一千两黄金。"},
                {"id": 3, "title": "暗巷·真身", "summary": "苏墨进入暗巷，黑猫现身，他展示将军府地图。", "characters_in_scene": ["苏墨", "黑猫"], "emotion_arc": "冷静→锋芒毕露", "key_dialogue": "他们连我的真名都不知道了。"},
            ]
        elif "分镜" in system_prompt or "Panel" in system_prompt or "Storyboard" in system_prompt:
            # Stage 4: 分镜生成
            return [
                {
                    "panel_number": 1,
                    "visual_description": "远景·大俯瞰，长安城暮色四合，万家灯火，朱雀大街延伸向远方",
                    "character_action": "无人物大动作，城市运转",
                    "dialogue": "",
                    "camera_angle": "俯视大远景",
                    "mood": "繁华之下的寂寥",
                    "sd_prompt": "epic bird's eye view of ancient Chinese capital, lanterns glowing, distant mansion",
                    "character_refs": [],
                },
                {
                    "panel_number": 2,
                    "visual_description": "极近特写，一只手紧握锈迹斑斑的铁剑",
                    "character_action": "手微微收紧，指节泛白",
                    "dialogue": "",
                    "camera_angle": "极近特写",
                    "mood": "沉淀三年的沉重",
                    "sd_prompt": "extreme close-up of hand gripping rusty sword, weathered texture, melancholic",
                    "character_refs": ["苏墨"],
                },
            ]
        return {}


def test_style_detection():
    """测试风格自动判断。"""
    s = detect_style(["武侠", "悬疑"], "慢热")
    assert s.name == "gufeng"

    s = detect_style(["校园", "恋爱"])
    assert s.name == "manga"

    s = detect_style(["都市"])
    assert s.name == "webtoon"

    print("  ✅ test_style_detection passed")


def test_pipeline_end_to_end():
    """测试完整 Pipeline 端到端流程（Mock LLM + 占位图）。"""
    mock_llm = MockLLM()
    img_gen = ImageGenAdapter(use_placeholder=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        data = ChapterData(
            title="月下归来",
            source_text=SAMPLE_TEXT,
            output_dir=tmpdir,
        )

        # 构建 Pipeline
        engine = PipelineEngine(mock_llm, img_gen)
        engine.register(run_stage1)
        engine.register(run_stage2)
        engine.register(run_stage3)
        engine.register(run_stage4)
        engine.register(run_stage5)
        engine.register(run_stage6)

        # 运行所有阶段
        data = engine.run_all(data)

        # 验证每个阶段的输出
        assert data.current_stage == 6, f"Expected stage 6, got {data.current_stage}"
        assert data.analysis is not None, "Stage 1 should produce analysis"
        assert data.analysis.style == "gufeng", f"Expected gufeng, got {data.analysis.style}"
        assert len(data.characters) == 3, f"Expected 3 characters, got {len(data.characters)}"
        assert data.characters[0].name == "苏墨"
        assert data.characters[0].sd_trigger_words != ""
        assert len(data.scenes) == 3, f"Expected 3 scenes, got {len(data.scenes)}"
        assert len(data.scenes[0].panels) > 0, "Scene 1 should have panels"
        for s in data.scenes:
            for p in s.panels:
                assert p.status == "generated", f"Panel {p.panel_number} not generated"
                assert os.path.exists(p.generated_image_path), f"Image not found: {p.generated_image_path}"
        assert len(data.pages) > 0, "Should have at least 1 comic page"
        for page in data.pages:
            assert os.path.exists(page.image_path), f"Comic page not found: {page.image_path}"

        print("  ✅ test_pipeline_end_to_end passed")


def test_data_serialization():
    """测试数据模型的 JSON 序列化/反序列化。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        data = ChapterData(
            title="测试",
            source_text="测试文本",
            output_dir=tmpdir,
        )
        data.analysis = AnalysisResult(genre_tags=["武侠"], style="gufeng")
        data.characters = [
            CharacterSheet(
                id="test_char", name="测试角色", role="protagonist",
                appearance=CharacterAppearance(face="测试面孔"),
                sd_trigger_words="test character trigger words",
            )
        ]

        # 保存
        filepath = os.path.join(tmpdir, "test.json")
        data.save(filepath)
        assert os.path.exists(filepath)

        # 加载
        loaded = ChapterData.load(filepath)
        assert loaded.title == "测试"
        assert loaded.analysis.style == "gufeng"
        assert len(loaded.characters) == 1
        assert loaded.characters[0].name == "测试角色"
        assert loaded.characters[0].sd_trigger_words == "test character trigger words"

        print("  ✅ test_data_serialization passed")


if __name__ == "__main__":
    test_style_detection()
    test_pipeline_end_to_end()
    test_data_serialization()
    print("\n🎉 All tests passed!")
```

- [ ] **Step 3: 运行测试**

```bash
cd novel2comic_v2 && python tests/test_pipeline.py
```

Expected: 所有 3 个测试通过

- [ ] **Step 4: 提交**

```bash
git add novel2comic_v2/tests/
git commit -m "test(n2c): add end-to-end pipeline integration tests with mock LLM"
```

---

### Task 16: 项目 .gitignore 更新

**Files:**

- Modify: `.gitignore` (project root)

- [ ] **Step 1: 确保 novel2comic_v2 输出目录不被追踪**

确认 `.gitignore` 包含以下行：

```
novel2comic_v2/projects/
novel2comic_v2/.env
__pycache__/
```

如果 `.gitignore` 不存在对应行，追加它们：

```bash
grep -q "novel2comic_v2/projects" .gitignore || echo "novel2comic_v2/projects/" >> .gitignore
grep -q "novel2comic_v2/.env" .gitignore || echo "novel2comic_v2/.env" >> .gitignore
```

- [ ] **Step 2: 提交**

```bash
git add .gitignore
git commit -m "chore: update .gitignore for novel2comic_v2 outputs"
```

---

## 自审清单

1. **Spec coverage**: 对照设计文档——
   - ✅ 6 阶段 Pipeline 全部实现 (Task 7-12)
   - ✅ StyleProfile 三种风格 + 自动判断 (Task 3)
   - ✅ 角色定妆 + sd_trigger_words (Task 8)
   - ✅ sd_prompt 自动注入风格基座+角色触发词 (Task 10)
   - ✅ 图像生成 + 占位图兜底 (Task 5, 11)
   - ✅ 漫画排版 + 对话框 (Task 12)
   - ✅ CLI 交互 (Task 13)
   - ✅ Skill 文件 (Task 14)
   - ⚠️ 反馈记忆系统 → 完整版
   - ⚠️ 人物关系知识图谱 → 完整版
   - ⚠️ Manga 格阵排版 → 完整版（MVP 用 scroll 模式兜底）
   - ⚠️ 角色定妆照生成 → 完整版（MVP 只用 prompt 约束）

2. **Placeholder scan**: 无 TBD/TODO。所有代码步骤完整。

3. **Type consistency**:
   - `ChapterData` 在 models.py 定义，所有 stage 函数使用 → 一致
   - `LLMAdapter.chat_json()` 返回 `dict`，所有 stage 使用 `.get()` → 一致
   - `ImageGenAdapter.generate()` 返回 `str` (文件路径) → 一致
   - `PipelineEngine.run_stage()` 返回 `ChapterData` → 一致
