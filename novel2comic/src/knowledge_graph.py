# -*- coding: utf-8 -*-
"""知识图谱提取引擎——从小说文本中提取结构化的人物关系图。"""

import json
from typing import Optional
from src.models import CharacterGraph, CharacterNode, RelationshipEdge


EXTRACTION_PROMPT = """你是一位专业的小说分析师。你需要从小说文本中提取人物关系知识图谱。

## 任务
分析以下文本，提取所有角色及其之间的关系。以 JSON 格式返回。

## 返回格式
{
  "nodes": [
    {
      "id": "英文名_小写_下划线",
      "name": "中文名",
      "role_type": "protagonist|antagonist|supporting|minor",
      "faction": "所属势力或阵营（如'将军府'、'江湖'、'无'）",
      "importance": 1-10的整数（主角10、主要配角7-9、次要配角4-6、路人1-3）,
      "status": "active|dead|missing|unknown",
      "description": "一句话描述这个角色是什么人"
    }
  ],
  "edges": [
    {
      "from_char": "角色A的name（必须与nodes中的name完全一致）",
      "to_char": "角色B的name",
      "relation_type": "血缘|爱情|友情|敌对|师徒|主仆|利用|同盟|陌生",
      "sub_type": "更具体的子类型（如'暗恋'、'杀父之仇'、'青梅竹马'、'背叛'、'上下级'等）",
      "intimacy": -10到+10的整数（-10=不共戴天, 0=陌生人, +10=生死相依）,
      "power_dynamic": "平等|A主导|B主导|互相制衡",
      "public_knowledge": true或false（这层关系其他人知道吗？）,
      "current_tension": "和谐|紧张|暧昧|一触即发|冷战",
      "shared_history": "两人共同的经历（一句话概括）"
    }
  ]
}

## 规则
1. 只提取文中实际出现或明确提到的角色
2. 只提取文中可以推断的关系，不要凭空创造
3. intimacy 从对话语气、互动距离、心理描写推断
4. 如果有角色外貌描写，写入 description
5. nodes 的 name 用中文原名，id 用英文"""


UPDATE_PROMPT = """你是一位专业的小说分析师。以下是已有的人物关系图谱，请根据新的章节内容更新它。

## 已有图谱
{existing_graph}

## 新章节内容
{chapter_text}

## 任务
分析新章节中的人物关系变化，返回 JSON：

{{
  "new_nodes": [...],      // 新出场的角色（格式同上）
  "new_edges": [...],      // 新的关系
  "updated_edges": [       // 变化的关系
    {{
      "from_char": "A",
      "to_char": "B",
      "changes": {{
        "intimacy": {{"old": -5, "new": -8, "reason": "A发现B是卧底"}},
        "current_tension": {{"old": "和谐", "new": "一触即发", "reason": "..."}}
      }}
    }}
  ]
}}

只返回有实际变化的数据。没有变化就返回空数组。"""


def extract_graph_from_text(
    text: str,
    openai_client,
    model: str = "deepseek-chat",
    temperature: float = 0.3,
) -> CharacterGraph:
    """从文本中提取人物关系知识图谱。

    Args:
        text: 小说文本（一章或多章）
        openai_client: OpenAI 兼容客户端
        model: LLM 模型名
        temperature: 生成温度

    Returns:
        CharacterGraph 实例
    """
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": f"请分析以下小说文本，提取人物关系知识图谱：\n\n{text[:8000]}"},
        ],
        temperature=temperature,
        timeout=120,
        max_tokens=4096,
    )

    content = response.choices[0].message.content or ""
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines)

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return CharacterGraph()

    graph = CharacterGraph()

    for nd in data.get("nodes", []):
        node = CharacterNode(
            id=nd.get("id", f"char_{len(graph.nodes):03d}"),
            name=nd.get("name", ""),
            role_type=nd.get("role_type", ""),
            faction=nd.get("faction", ""),
            importance=nd.get("importance", 5),
            status=nd.get("status", "active"),
            description=nd.get("description", ""),
        )
        graph.nodes.append(node)

    for ed in data.get("edges", []):
        edge = RelationshipEdge(
            from_char=ed.get("from_char", ""),
            to_char=ed.get("to_char", ""),
            relation_type=ed.get("relation_type", ""),
            sub_type=ed.get("sub_type", ""),
            intimacy=ed.get("intimacy", 0),
            power_dynamic=ed.get("power_dynamic", "平等"),
            public_knowledge=ed.get("public_knowledge", True),
            current_tension=ed.get("current_tension", "和谐"),
            shared_history=ed.get("shared_history", ""),
        )
        graph.add_edge(edge)

    return graph


def update_graph_with_chapter(
    graph: CharacterGraph,
    chapter_text: str,
    chapter_index: int,
    openai_client,
    model: str = "deepseek-chat",
) -> CharacterGraph:
    """用新章节更新已有的知识图谱。

    Args:
        graph: 现有的图谱
        chapter_text: 新章节文本
        chapter_index: 章节编号
        openai_client: LLM 客户端
        model: 模型名

    Returns:
        更新后的图谱（直接修改传入的 graph）
    """
    existing_summary = json.dumps({
        "nodes": [{"name": n.name, "role": n.role_type, "faction": n.faction}
                   for n in graph.nodes],
        "edges": [{"from": e.from_char, "to": e.to_char, "type": e.relation_type,
                    "intimacy": e.intimacy, "tension": e.current_tension}
                   for e in graph.edges],
    }, ensure_ascii=False, indent=2)

    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": UPDATE_PROMPT.format(
                existing_graph=existing_summary,
                chapter_text=chapter_text[:6000],
            )},
            {"role": "user", "content": "请分析新章节并返回图谱更新。"},
        ],
        temperature=0.3,
        timeout=120,
        max_tokens=4096,
    )

    content = response.choices[0].message.content or ""
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines)

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return graph

    # 添加新节点
    for nd in data.get("new_nodes", []):
        if not graph.get_node(nd.get("name", "")):
            node = CharacterNode(
                id=nd.get("id", f"char_{len(graph.nodes):03d}"),
                name=nd.get("name", ""),
                role_type=nd.get("role_type", ""),
                faction=nd.get("faction", ""),
                importance=nd.get("importance", 5),
                status=nd.get("status", "active"),
                first_appearance_chapter=chapter_index,
                description=nd.get("description", ""),
            )
            graph.nodes.append(node)

    # 添加新边
    for ed in data.get("new_edges", []):
        edge = RelationshipEdge(
            from_char=ed.get("from_char", ""),
            to_char=ed.get("to_char", ""),
            relation_type=ed.get("relation_type", ""),
            sub_type=ed.get("sub_type", ""),
            intimacy=ed.get("intimacy", 0),
            power_dynamic=ed.get("power_dynamic", "平等"),
            public_knowledge=ed.get("public_knowledge", True),
            current_tension=ed.get("current_tension", "和谐"),
            shared_history=ed.get("shared_history", ""),
            established_chapter=chapter_index,
        )
        graph.add_edge(edge)

    # 更新已有边
    for upd in data.get("updated_edges", []):
        edge = graph.get_edge(upd.get("from_char", ""), upd.get("to_char", ""))
        if edge:
            changes = upd.get("changes", {})
            for field, change in changes.items():
                if hasattr(edge, field):
                    setattr(edge, field, change.get("new", getattr(edge, field)))

    graph.last_updated_chapter = chapter_index
    return graph


def graph_to_context(graph: CharacterGraph) -> str:
    """将知识图谱格式化为 LLM prompt 可用的文本上下文。"""
    if not graph or not graph.nodes:
        return ""

    lines = ["[人物关系知识图谱]"]

    # 角色列表
    lines.append("\n## 角色")
    for node in sorted(graph.nodes, key=lambda n: -n.importance):
        status_mark = {"active": "", "dead": "[已死]", "missing": "[失踪]", "unknown": "[未知]"}.get(node.status, "")
        lines.append(
            f"- {node.name} [{node.role_type}] {status_mark}"
            + (f" | {node.faction}" if node.faction else "")
            + (f" | {node.description}" if node.description else "")
        )

    # 关系网络
    lines.append("\n## 关系网络")
    for edge in graph.edges:
        intimacy_bar = "█" * abs(edge.intimacy) if edge.intimacy >= 0 else "▓" * abs(edge.intimacy)
        public = "" if edge.public_knowledge else "[隐藏]"
        lines.append(
            f"- {edge.from_char} ←→ {edge.to_char}: {edge.relation_type}"
            + (f"({edge.sub_type})" if edge.sub_type else "")
            + f" | 亲密度:{edge.intimacy:+d} {intimacy_bar}"
            + f" | {edge.power_dynamic} | {edge.current_tension} {public}"
            + (f" | {edge.shared_history}" if edge.shared_history else "")
        )

    return "\n".join(lines)
