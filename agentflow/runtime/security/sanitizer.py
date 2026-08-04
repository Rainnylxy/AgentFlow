"""Parameter sanitizers for dangerous input patterns.

Each sanitizer maps a key string to a transformation function.
A tool's SecurityPolicy.sensitive_params lists which sanitizer keys
to apply to each named parameter before execution.
"""

from __future__ import annotations

import re
from typing import Callable


def _sanitize_path(value: str) -> str:
    """Strip path traversal sequences (../, ..\\, null bytes)."""
    if not isinstance(value, str):
        return value
    v = value.replace("\x00", "")
    v = re.sub(r"\.\.[/\\]", "", v)
    v = re.sub(r"[/\\]+", "/", v)
    v = v.strip("/")
    return v


def _sanitize_sql(value: str) -> str:
    """Remove common SQL injection metacharacters.

    Does NOT replace proper parameterized queries — this is a
    defense-in-depth layer for tools that construct dynamic SQL.
    """
    if not isinstance(value, str):
        return value
    # Strip comment markers and statement terminators
    v = re.sub(r"--.*$", "", value, flags=re.MULTILINE)
    v = v.replace("';", "")
    return v


def _sanitize_command(value: str) -> str:
    """Remove shell metacharacters that enable command chaining."""
    if not isinstance(value, str):
        return value
    # Strip ;, &, |, `, $, newlines — typical injection vectors
    v = re.sub(r"[;&|`$]", "", value)
    v = v.replace("\n", "").replace("\r", "")
    return v


def _sanitize_html(value: str) -> str:
    """Escape HTML entities to prevent XSS in rendered output."""
    if not isinstance(value, str):
        return value
    v = value.replace("&", "&amp;")
    v = v.replace("<", "&lt;")
    v = v.replace(">", "&gt;")
    v = v.replace('"', "&quot;")
    v = v.replace("'", "&#x27;")
    return v


# Registry of sanitizer functions keyed by name.
_REGISTRY: dict[str, Callable[[str], str]] = {
    "path": _sanitize_path,
    "sql": _sanitize_sql,
    "command": _sanitize_command,
    "html": _sanitize_html,
}


def get_sanitizer(name: str) -> Callable[[str], str] | None:
    return _REGISTRY.get(name)


def sanitize(params: dict, sensitive: list[str]) -> dict:
    """Apply registered sanitizers to sensitive params.

    Each entry in *sensitive* is either a bare key (applies a sanitizer
    of the same name) or a ``"key:sanitizer"`` pair.

    Returns a new dict with sanitized values.
    """
    cleaned = dict(params)
    for spec in sensitive:
        if ":" in spec:
            key, sanitizer_name = spec.split(":", 1)
        else:
            key = spec
            sanitizer_name = spec
        fn = _REGISTRY.get(sanitizer_name)
        if fn is None:
            continue
        if key in cleaned and isinstance(cleaned[key], str):
            cleaned[key] = fn(cleaned[key])
    return cleaned
