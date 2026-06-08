# -*- coding: utf-8 -*-
"""Novel RAG——将小说内容索引为可检索的知识库。

功能：
- 分块 (chunk): 按段落切分，每块带章节/位置元数据
- 嵌入 (embed): 调用 LLM API 的 embedding 端点
- 检索 (search): 余弦相似度匹配，返回最相关段落
- 角色检索: 按角色名查找所有出场段落

存储: JSON 文件（chunks + embeddings），放在项目目录下。
"""

import os
import json
import math
from typing import Optional


class NovelRAG:
    """小说知识库——chunk + embed + search。"""

    def __init__(self, storage_dir: str):
        self.storage_dir = storage_dir
        self.chunks: list[dict] = []        # [{id, text, chapter, chapter_title, chars_mentioned, ...}]
        self.embeddings: list[list[float]] = []  # 与 chunks 一一对应
        self._indexed = False

    # ================================================================
    # 索引
    # ================================================================

    def index_novel(
        self,
        chapters: list,          # list[ChapterInfo]
        openai_client,           # OpenAI 同步客户端
        model: str = "text-embedding-3-small",
        batch_size: int = 20,
    ) -> int:
        """将整本小说分块并嵌入索引。

        Args:
            chapters: ChapterInfo 列表
            openai_client: OpenAI 兼容客户端
            model: embedding 模型名
            batch_size: 每批嵌入的块数

        Returns:
            索引的块总数
        """
        # 1. 分块
        self.chunks = []
        for chapter in chapters:
            paragraphs = self._split_chapter(chapter.content)
            for i, para in enumerate(paragraphs):
                if len(para.strip()) < 10:  # 跳过太短的段落
                    continue
                chars = self._extract_characters(para, chapter)
                self.chunks.append({
                    "id": f"ch{chapter.index:04d}_{i:03d}",
                    "text": para,
                    "chapter_index": chapter.index,
                    "chapter_title": chapter.title,
                    "position": i,
                    "chars_mentioned": chars,
                })

        if not self.chunks:
            return 0

        # 2. 批量嵌入
        self.embeddings = []
        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i:i + batch_size]
            texts = [c["text"] for c in batch]
            try:
                resp = openai_client.embeddings.create(
                    model=model,
                    input=texts,
                )
                for item in resp.data:
                    self.embeddings.append(item.embedding)
            except Exception as e:
                # embedding API 不可用时回退为伪嵌入（长度向量）
                print(f"  [RAG] Embedding API failed ({e}), using fallback")
                for t in texts:
                    self.embeddings.append(self._fallback_embed(t))

        self._indexed = True
        return len(self.chunks)

    def _split_chapter(self, text: str) -> list[str]:
        """按段落分块（双换行或单换行分隔）。"""
        # 先按双换行分，再按单换行分长段落
        raw = text.split("\n\n")
        result = []
        for block in raw:
            block = block.strip()
            if not block:
                continue
            if len(block) > 500:
                # 长段落再按单换行细分
                sub = block.split("\n")
                for s in sub:
                    s = s.strip()
                    if s:
                        result.append(s)
            else:
                result.append(block)
        return result

    def _extract_characters(self, text: str, chapter) -> list[str]:
        """从段落中提取提及的角色名（简单关键词匹配）。"""
        # 这里用最简单的匹配，后续可升级为 NER
        # 如果没有角色库，返回空列表
        return []  # 由外部注入角色列表后重新计算

    # ================================================================
    # 检索
    # ================================================================

    def search(
        self,
        query: str,
        top_k: int = 5,
        openai_client=None,
        model: str = "text-embedding-3-small",
    ) -> list[dict]:
        """检索与查询最相关的段落。

        Args:
            query: 查询文本
            top_k: 返回前 K 个结果
            openai_client: 用于生成查询嵌入
            model: embedding 模型

        Returns:
            [{chunk, score, chapter_index, chapter_title}, ...]
        """
        if not self.chunks or not self.embeddings:
            return []

        # 生成查询嵌入
        try:
            resp = openai_client.embeddings.create(model=model, input=[query])
            query_emb = resp.data[0].embedding
        except Exception:
            query_emb = self._fallback_embed(query)

        # 计算余弦相似度
        scores = []
        for i, emb in enumerate(self.embeddings):
            sim = self._cosine_similarity(query_emb, emb)
            scores.append((sim, i))

        scores.sort(key=lambda x: x[0], reverse=True)

        results = []
        for sim, idx in scores[:top_k]:
            if sim < 0.1:  # 相似度太低，跳过
                continue
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk["id"],
                "text": chunk["text"],
                "score": round(sim, 4),
                "chapter_index": chunk["chapter_index"],
                "chapter_title": chunk["chapter_title"],
                "chars_mentioned": chunk["chars_mentioned"],
            })

        return results

    def search_by_character(
        self,
        character_name: str,
        top_k: int = 10,
    ) -> list[dict]:
        """检索包含指定角色的所有段落（简单子串匹配 + 嵌入排序）。"""
        if not self.chunks:
            return []

        # 先用子串匹配粗筛
        candidates = []
        for i, chunk in enumerate(self.chunks):
            if character_name in chunk["text"]:
                candidates.append(i)

        if not candidates:
            return []

        # 返回匹配的段落（按章节顺序）
        results = []
        for idx in candidates[:top_k]:
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk["id"],
                "text": chunk["text"],
                "chapter_index": chunk["chapter_index"],
                "chapter_title": chunk["chapter_title"],
            })

        return results

    def search_across_chapters(
        self,
        query: str,
        chapter_range=None,  # Optional[tuple[int, int]]
        top_k: int = 5,
        openai_client=None,
        model: str = "text-embedding-3-small",
    ) -> list[dict]:
        """跨章节检索，可选限制章节范围。

        Args:
            query: 查询文本
            chapter_range: (start, end) 章节范围，None 表示全书
            top_k: 返回数量
            openai_client: embedding 客户端
            model: embedding 模型
        """
        results = self.search(query, top_k=top_k * 2, openai_client=openai_client, model=model)

        if chapter_range:
            start, end = chapter_range
            results = [r for r in results if start <= r["chapter_index"] <= end]

        return results[:top_k]

    # ================================================================
    # 持久化
    # ================================================================

    def save(self, filepath: str):
        """保存索引到 JSON 文件。"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "chunks": self.chunks,
            "embeddings": self.embeddings,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, filepath: str) -> bool:
        """从 JSON 文件加载索引。"""
        if not os.path.exists(filepath):
            return False
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.chunks = data.get("chunks", [])
        self.embeddings = data.get("embeddings", [])
        self._indexed = len(self.chunks) > 0
        return self._indexed

    @property
    def is_indexed(self) -> bool:
        return self._indexed

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    # ================================================================
    # 内部工具
    # ================================================================

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """计算两个向量的余弦相似度。"""
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _fallback_embed(text: str, dim: int = 256) -> list[float]:
        """回退伪嵌入：基于字符哈希生成向量（embedding API 不可用时使用）。"""
        # 用简单哈希生成确定性的伪嵌入——至少保证相同文本匹配
        import hashlib
        result = []
        for i in range(dim):
            h = hashlib.md5(f"{text}_{i}".encode()).digest()
            # 将 16 字节哈希转为 16 个浮点数（归一化到 [0, 1]）
            val = sum(h) / (256 * 16)
            result.append(val)
        # 归一化
        norm = math.sqrt(sum(x * x for x in result))
        if norm > 0:
            result = [x / norm for x in result]
        return result


# ================================================================
# 便捷函数
# ================================================================

def create_rag_for_novel(novel_dir: str) -> NovelRAG:
    """为小说项目目录创建 RAG 实例。"""
    return NovelRAG(novel_dir)


def build_context_from_search(
    rag: NovelRAG,
    query: str,
    openai_client,
    model: str = "text-embedding-3-small",
    top_k: int = 5,
    max_chars: int = 2000,
) -> str:
    """从 RAG 检索结果构建上下文文本（用于注入 LLM prompt）。

    Args:
        rag: NovelRAG 实例
        query: 检索查询
        openai_client: embedding 客户端
        model: embedding 模型
        top_k: 检索块数
        max_chars: 上下文最大字符数

    Returns:
        格式化的上下文字符串
    """
    results = rag.search(query, top_k=top_k, openai_client=openai_client, model=model)
    if not results:
        return ""

    lines = ["[从全书检索到的相关上下文]"]
    total = 0
    for r in results:
        snippet = r["text"][:300]
        line = f"[第{r['chapter_index']}章《{r['chapter_title']}》相似度:{r['score']:.2f}] {snippet}"
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line)

    return "\n".join(lines)


def build_character_context(
    rag: NovelRAG,
    character_name: str,
    max_chunks: int = 8,
) -> str:
    """从 RAG 检索角色相关上下文。

    Args:
        rag: NovelRAG 实例
        character_name: 角色名
        max_chunks: 最大返回块数

    Returns:
        角色出场记录文本
    """
    results = rag.search_by_character(character_name, top_k=max_chunks)
    if not results:
        return ""

    lines = [f"[角色 '{character_name}' 在全书中的出场记录]"]
    for r in results:
        lines.append(f"- 第{r['chapter_index']}章《{r['chapter_title']}》: {r['text'][:200]}")

    return "\n".join(lines)
