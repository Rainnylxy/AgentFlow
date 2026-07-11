"""Reference — 参考卡：跨 run 持久、永不裁剪的上下文信息。

与 WorkingMemory（History）不同，Reference 中的消息始终驻留在
上下文窗口中，不参与滑动窗口裁剪。适用场景：
- 角色设定、世界观、文风等写作参考
- 项目约定、用户偏好等长期上下文
- 应用层动态更新的摘要/概要

用法:
    ref = Reference()
    ref.set("characters", "张三-剑客-30岁")
    ref.set("style", "古风武侠，冷峻克制")

    # 生成可喂给 LLM 的消息列表
    messages = ref.to_messages()
    # → [{"role": "system", "content": "[Reference: characters]\n张三-剑客-30岁"},
    #    {"role": "system", "content": "[Reference: style]\n古风武侠，冷峻克制"}]
"""

from __future__ import annotations

from typing import Optional


class Reference:
    """键值对形式的参考卡，所有条目永不裁剪。

    key 用于标识和更新（如 "characters", "summary", "style"），
    content 为实际注入 LLM 上下文的文本。
    """

    def __init__(self):
        self._entries: dict[str, str] = {}

    def set(self, key: str, content: str) -> None:
        """设置或更新一个参考条目。"""
        self._entries[key] = content

    def get(self, key: str) -> Optional[str]:
        """获取指定 key 的参考内容。"""
        return self._entries.get(key)

    def remove(self, key: str) -> None:
        """删除一个参考条目。"""
        self._entries.pop(key, None)

    def clear(self) -> None:
        """清空所有参考条目。"""
        self._entries.clear()

    def to_messages(self) -> list[dict]:
        """将参考卡转为 system 消息列表，供策略注入 LLM 上下文。

        Returns:
            list[dict]: 每个条目一条 system 消息，
                        role="system"，content 格式为 "[Reference: {key}]\n{content}"
        """
        return [
            {"role": "system", "content": f"[Reference: {key}]\n{content}"}
            for key, content in self._entries.items()
        ]

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def __repr__(self) -> str:
        keys = list(self._entries.keys())
        return f"Reference(keys={keys})"
