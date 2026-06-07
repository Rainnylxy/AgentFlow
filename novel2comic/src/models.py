# -*- coding: utf-8 -*-
"""Novel2Comic V2 数据模型——所有 dataclass 定义 + JSON 序列化。"""

import os
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime
import json


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
    id: str
    name: str
    role: str
    appearance: CharacterAppearance = field(default_factory=CharacterAppearance)
    reference_image_path: str = ""
    sd_trigger_words: str = ""
    personality_notes: str = ""
    status: str = "draft"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["appearance"] = asdict(self.appearance)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CharacterSheet":
        appearance = CharacterAppearance(**d.pop("appearance", {}))
        return cls(appearance=appearance, **d)


@dataclass
class AnalysisResult:
    genre_tags: list[str] = field(default_factory=list)
    style: str = "auto"
    tone: list[str] = field(default_factory=list)
    era: str = ""
    pace: str = ""
    characters_preview: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AnalysisResult":
        return cls(**d)


@dataclass
class Panel:
    panel_number: int = 0
    visual_description: str = ""
    character_action: str = ""
    dialogue: str = ""
    camera_angle: str = ""
    mood: str = ""
    sd_prompt: str = ""
    character_refs: list[str] = field(default_factory=list)
    generated_image_path: str = ""
    status: str = "pending"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Panel":
        return cls(**d)


@dataclass
class Scene:
    id: int = 0
    title: str = ""
    summary: str = ""
    characters_in_scene: list[str] = field(default_factory=list)
    emotion_arc: str = ""
    key_dialogue: str = ""
    panels: list[Panel] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["panels"] = [p.to_dict() for p in self.panels]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Scene":
        panels = [Panel.from_dict(p) for p in d.pop("panels", [])]
        return cls(panels=panels, **d)


@dataclass
class ComicPage:
    page_number: int = 0
    image_path: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ComicPage":
        return cls(**d)


@dataclass
class ChapterInfo:
    """章节元数据——从小说中解析出的章节信息。"""
    index: int = 0                  # 第几章 (1-based)
    title: str = ""                 # 章节标题
    content: str = ""               # 章节正文
    word_count: int = 0             # 字数
    status: str = "pending"         # "pending" | "generating" | "completed"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ChapterInfo":
        return cls(**d)


@dataclass
class Novel:
    """全书——顶层数据模型，包含章节列表和跨章节共享的角色库。"""
    title: str = ""                          # 书名
    file_path: str = ""                      # 原始文件路径
    chapters: list[ChapterInfo] = field(default_factory=list)  # 章节列表
    characters: list[CharacterSheet] = field(default_factory=list)  # 全书角色库（跨章节共享）
    style_profile: Optional[StyleProfile] = None  # 全书风格（首次分析后锁定）
    current_chapter_index: int = 0           # 当前选中的章节 (1-based)
    output_dir: str = ""

    @property
    def current_chapter(self) -> Optional[ChapterInfo]:
        """当前选中的章节。"""
        for ch in self.chapters:
            if ch.index == self.current_chapter_index:
                return ch
        return None

    @property
    def total_chapters(self) -> int:
        return len(self.chapters)

    def get_characters_by_name(self, name: str) -> list[CharacterSheet]:
        """按名称查找角色（支持模糊匹配）。"""
        return [c for c in self.characters if c.name == name]

    def has_character(self, name: str) -> bool:
        return any(c.name == name for c in self.characters)

    def add_characters(self, new_chars: list[CharacterSheet]):
        """添加角色到全书库（同名跳过）。"""
        existing = {c.name for c in self.characters}
        for char in new_chars:
            if char.name not in existing:
                self.characters.append(char)
                existing.add(char.name)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "file_path": self.file_path,
            "chapters": [ch.to_dict() for ch in self.chapters],
            "characters": [c.to_dict() for c in self.characters],
            "style_profile": self.style_profile.to_dict() if self.style_profile else None,
            "current_chapter_index": self.current_chapter_index,
            "output_dir": self.output_dir,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Novel":
        novel = cls(
            title=d.get("title", ""),
            file_path=d.get("file_path", ""),
            current_chapter_index=d.get("current_chapter_index", 0),
            output_dir=d.get("output_dir", ""),
        )
        novel.chapters = [ChapterInfo.from_dict(ch) for ch in d.get("chapters", [])]
        novel.characters = [CharacterSheet.from_dict(c) for c in d.get("characters", [])]
        if d.get("style_profile"):
            novel.style_profile = StyleProfile.from_dict(d["style_profile"])
        return novel

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "Novel":
        with open(filepath, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


@dataclass
class ChapterData:
    """Pipeline 数据总线——单章生成的共享状态（6 阶段）。"""
    title: str = ""
    source_text: str = ""
    analysis: Optional[AnalysisResult] = None
    characters: list[CharacterSheet] = field(default_factory=list)
    scenes: list[Scene] = field(default_factory=list)
    pages: list[ComicPage] = field(default_factory=list)
    style_profile: Optional[StyleProfile] = None
    current_stage: int = 0
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
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "ChapterData":
        with open(filepath, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
