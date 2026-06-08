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


# ============================================================
# 知识图谱
# ============================================================

@dataclass
class CharacterNode:
    """角色节点。"""
    id: str = ""                              # 唯一标识
    name: str = ""                            # 中文名
    role_type: str = ""                       # "protagonist" | "antagonist" | "supporting" | "minor"
    faction: str = ""                         # 所属势力/阵营
    importance: int = 5                       # 1-10 重要程度
    first_appearance_chapter: int = 0
    status: str = "active"                    # "active" | "dead" | "missing" | "unknown"
    description: str = ""                     # 一句话描述

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CharacterNode":
        return cls(**d)


@dataclass
class RelationshipEdge:
    """关系边。"""
    from_char: str = ""                       # → CharacterNode.id
    to_char: str = ""
    relation_type: str = ""                   # "血缘"|"爱情"|"友情"|"敌对"|"师徒"|"主仆"|"利用"|"同盟"
    sub_type: str = ""                        # "暗恋"|"杀父之仇"|"青梅竹马"|"背叛"|...
    intimacy: int = 0                         # -10(不共戴天) ~ +10(生死相依)
    power_dynamic: str = "平等"               # "平等"|"A主导"|"B主导"|"互相制衡"
    public_knowledge: bool = True             # 关系是否公开
    current_tension: str = "和谐"             # "和谐"|"紧张"|"暧昧"|"一触即发"|"冷战"
    shared_history: str = ""                  # 共同经历摘要
    established_chapter: int = 0              # 关系建立的章

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RelationshipEdge":
        return cls(**d)


@dataclass
class RelationEvent:
    """关系变化事件——追踪关系随时间演变。"""
    chapter: int = 0
    from_char: str = ""
    to_char: str = ""
    field: str = ""                           # 变化的字段
    old_value: str = ""
    new_value: str = ""
    trigger_event: str = ""                   # 触发事件描述
    timestamp: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RelationEvent":
        return cls(**d)


@dataclass
class CharacterGraph:
    """人物关系知识图谱——基于 NetworkX。

    内部使用 networkx.Graph 存储节点和边，
    对外保持兼容的 API，同时暴露图算法（最短路径、中心度等）。
    """
    last_updated_chapter: int = 0
    timeline: list[RelationEvent] = field(default_factory=list)

    def __post_init__(self):
        import networkx as nx
        self._g = nx.Graph()

    # ================================================================
    # 节点操作
    # ================================================================

    def get_node(self, name: str) -> Optional[CharacterNode]:
        if name not in self._g:
            return None
        attrs = self._g.nodes[name]
        return CharacterNode(
            id=attrs.get("id", ""), name=name,
            role_type=attrs.get("role_type", ""),
            faction=attrs.get("faction", ""),
            importance=attrs.get("importance", 5),
            first_appearance_chapter=attrs.get("first_appearance_chapter", 0),
            status=attrs.get("status", "active"),
            description=attrs.get("description", ""),
        )

    def get_or_create_node(self, name: str) -> CharacterNode:
        node = self.get_node(name)
        if not node:
            node = CharacterNode(id=f"char_{len(self._g):03d}", name=name)
            self._add_node(node)
        return node

    def _add_node(self, node: CharacterNode):
        self._g.add_node(node.name,
            id=node.id, role_type=node.role_type, faction=node.faction,
            importance=node.importance, first_appearance_chapter=node.first_appearance_chapter,
            status=node.status, description=node.description,
        )

    @property
    def nodes(self) -> list[CharacterNode]:
        return [self.get_node(n) for n in self._g.nodes]

    @property
    def node_count(self) -> int:
        return self._g.number_of_nodes()

    # ================================================================
    # 边操作
    # ================================================================

    def get_edges(self, name: str) -> list[RelationshipEdge]:
        edges = []
        for neighbor in self._g.neighbors(name):
            edge = self.get_edge(name, neighbor)
            if edge:
                edges.append(edge)
        return edges

    def get_edge(self, a: str, b: str) -> Optional[RelationshipEdge]:
        if not self._g.has_edge(a, b):
            return None
        data = self._g.edges[a, b]
        return RelationshipEdge(
            from_char=a, to_char=b,
            relation_type=data.get("relation_type", ""),
            sub_type=data.get("sub_type", ""),
            intimacy=data.get("intimacy", 0),
            power_dynamic=data.get("power_dynamic", "平等"),
            public_knowledge=data.get("public_knowledge", True),
            current_tension=data.get("current_tension", "和谐"),
            shared_history=data.get("shared_history", ""),
            established_chapter=data.get("established_chapter", 0),
        )

    def add_edge(self, edge: RelationshipEdge):
        # 确保两端节点存在
        if edge.from_char not in self._g:
            self._g.add_node(edge.from_char)
        if edge.to_char not in self._g:
            self._g.add_node(edge.to_char)

        existing = self.get_edge(edge.from_char, edge.to_char)
        if existing:
            for field_name in ["relation_type", "sub_type", "intimacy", "power_dynamic",
                               "public_knowledge", "current_tension", "shared_history"]:
                new_val = getattr(edge, field_name, None)
                default_map = {"intimacy": 0, "power_dynamic": "平等", "current_tension": "和谐", "public_knowledge": True}
                default = default_map.get(field_name)
                if new_val not in (None, "", default):
                    old_val = getattr(existing, field_name)
                    if str(old_val) != str(new_val):
                        self.timeline.append(RelationEvent(
                            chapter=edge.established_chapter,
                            from_char=edge.from_char, to_char=edge.to_char,
                            field=field_name, old_value=str(old_val),
                            new_value=str(new_val),
                        ))
            # 直接覆盖所有属性
        self._g.add_edge(edge.from_char, edge.to_char,
            relation_type=edge.relation_type, sub_type=edge.sub_type,
            intimacy=edge.intimacy, power_dynamic=edge.power_dynamic,
            public_knowledge=edge.public_knowledge, current_tension=edge.current_tension,
            shared_history=edge.shared_history, established_chapter=edge.established_chapter,
        )

    @property
    def edges(self) -> list[RelationshipEdge]:
        result = []
        for a, b in self._g.edges:
            edge = self.get_edge(a, b)
            if edge:
                result.append(edge)
        return result

    @property
    def edge_count(self) -> int:
        return self._g.number_of_edges()

    # ================================================================
    # 图算法（NetworkX 提供）
    # ================================================================

    def shortest_path(self, a: str, b: str) -> Optional[list[str]]:
        """两角色之间的最短关系路径。"""
        import networkx as nx
        try:
            return nx.shortest_path(self._g, a, b)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def centrality_ranking(self, top_k: int = 10) -> list[tuple[str, float]]:
        """角色中心度排名（度中心性）。"""
        import networkx as nx
        dc = nx.degree_centrality(self._g)
        return sorted(dc.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def faction_groups(self) -> dict[str, list[str]]:
        """按阵营分组角色。"""
        groups: dict[str, list[str]] = {}
        for node in self.nodes:
            faction = node.faction or "无阵营"
            if faction not in groups:
                groups[faction] = []
            groups[faction].append(node.name)
        return groups

    def enemy_pairs(self) -> list[tuple[str, str]]:
        """列出所有敌对关系。"""
        return [(a, b) for a, b, d in self._g.edges(data=True)
                if d.get("relation_type") == "敌对"]

    def hidden_relations(self) -> list[tuple[str, str, str]]:
        """列出所有隐藏关系。"""
        return [(a, b, d.get("relation_type", ""))
                for a, b, d in self._g.edges(data=True)
                if not d.get("public_knowledge", True)]

    def intimacy_ranking(self) -> list[tuple[str, str, int]]:
        """按亲密度排序的关系列表。"""
        pairs = [(a, b, d.get("intimacy", 0)) for a, b, d in self._g.edges(data=True)]
        return sorted(pairs, key=lambda x: abs(x[2]), reverse=True)

    def story_path(self, char: str, max_depth: int = 2) -> dict:
        """角色的关系子图（用于故事线查看）。"""
        import networkx as nx
        neighbors = {}
        for depth in range(1, max_depth + 1):
            for other in self._g.nodes:
                if other == char:
                    continue
                try:
                    path = nx.shortest_path(self._g, char, other)
                    if len(path) - 1 == depth:
                        neighbors[other] = path
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
        return neighbors

    # ================================================================
    # 分镜指导
    # ================================================================

    def get_storyboard_hints(self, char_a: str, char_b: str) -> str:
        """根据两个角色的关系生成分镜指导提示。"""
        edge = self.get_edge(char_a, char_b)
        if not edge:
            return ""

        hints = []
        if edge.intimacy >= 7:
            hints.append("两人亲近，同框时距离近，用双人中近景，眼神交流，柔和光线")
        elif edge.intimacy <= -7:
            hints.append("两人敌对，同框时用对峙构图、低角度仰拍、特写眼神交锋、sd_prompt加'dramatic shadows'")

        if edge.power_dynamic == "A主导":
            hints.append(f"{edge.from_char}是上位者→仰拍显高大, {edge.to_char}俯拍显弱小")
        elif edge.power_dynamic == "B主导":
            hints.append(f"{edge.to_char}是上位者→仰拍显高大, {edge.from_char}俯拍显弱小")

        if not edge.public_knowledge:
            hints.append("关系隐藏→公开场合两人站远、表情克制、只在对视瞬间流露微表情")

        tension_map = {
            "暧昧": "避免直视、侧脸和偷看视角、sd_prompt加'shy glance, soft focus'",
            "紧张": "身体语言僵硬、避免眼神接触、画面留白营造窒息感",
            "一触即发": "动作预备姿态、面部紧绷、sd_prompt加'tense atmosphere, ready to strike'",
            "冷战": "背对背站位、各自看向不同方向、中间留空",
        }
        if edge.current_tension in tension_map:
            hints.append(tension_map[edge.current_tension])

        type_hints = {
            "爱情": "关注手部细节和微表情, sd_prompt加'romantic atmosphere'",
            "敌对": "多用斜线构图和速度线, sd_prompt加'confrontation'",
            "师徒": "A略高于B的站位、B带敬意的眼神",
        }
        if edge.relation_type in type_hints:
            hints.append(type_hints[edge.relation_type])

        return " | ".join(hints)

    # ================================================================
    # 序列化
    # ================================================================

    def to_dict(self) -> dict:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "timeline": [t.to_dict() for t in self.timeline],
            "last_updated_chapter": self.last_updated_chapter,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CharacterGraph":
        graph = cls(last_updated_chapter=d.get("last_updated_chapter", 0))
        for nd in d.get("nodes", []):
            node = CharacterNode.from_dict(nd)
            graph._add_node(node)
        for ed in d.get("edges", []):
            edge = RelationshipEdge.from_dict(ed)
            graph._g.add_node(edge.from_char)
            graph._g.add_node(edge.to_char)
            graph.add_edge(edge)
        graph.timeline = [RelationEvent.from_dict(t) for t in d.get("timeline", [])]
        return graph


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
    character_graph: Optional[CharacterGraph] = None  # 人物关系知识图谱
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
            "character_graph": self.character_graph.to_dict() if self.character_graph else None,
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
        if d.get("character_graph"):
            novel.character_graph = CharacterGraph.from_dict(d["character_graph"])
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
