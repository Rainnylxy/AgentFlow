"""Handoff protocol — standardized data structures for agent-to-agent task transfer.

HandoffRequest and RouteResult are plain dataclasses. parse_handoff_block()
extracts HANDOFF markers from text using regex. This module is consumed by
RoutingStrategy (Task 3).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class HandoffRequest:
    """Signals that an expert agent cannot complete the task and suggests
    another agent who can help.

    All fields have defaults — a minimal HandoffRequest with just *reason*
    is valid.
    """

    reason: str = ""
    #: Natural language description of who can help (NOT agent_id).
    suggested_agent: str = ""
    #: Why this suggestion makes sense.
    suggested_since: str = ""
    #: What the expert already figured out.
    partial_result: str = ""


@dataclass
class RouteResult:
    """The result of routing a task to an agent.

    When *handoff* is None the task is considered complete. When *handoff*
    is set the routing layer should re-evaluate and route to a different
    agent.
    """

    agent_id: str
    output: str = ""
    tool_calls: list = field(default_factory=list)
    handoff: HandoffRequest | None = None


def parse_handoff_block(text: str) -> HandoffRequest | None:
    """Extract a HANDOFF block from *text* and return a HandoffRequest.

    Expected format::

        ---HANDOFF---
        reason: <why you can't handle this>
        suggest: <natural language description of who can>
        context: <what the next expert should know>
        ---END---

    Returns *HandoffRequest* if a valid block is found, or *None* when no
    block is present or the block is malformed (missing ``---END---`` or
    missing the ``reason`` field). Extra whitespace in field values is
    stripped. Only ``reason`` is required; ``suggest`` and ``context`` are
    optional.
    """
    match = re.search(
        r'---HANDOFF---\s*\n(.*?)---END---',
        text,
        re.DOTALL,
    )
    if not match:
        return None

    content = match.group(1)
    fields: dict[str, str] = {}

    for line in content.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        fields[key] = value

    if "reason" not in fields:
        return None

    return HandoffRequest(
        reason=fields.get("reason", ""),
        suggested_agent=fields.get("suggest", ""),
        partial_result=fields.get("context", ""),
    )
