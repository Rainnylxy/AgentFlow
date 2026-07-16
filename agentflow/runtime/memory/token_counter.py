"""Token 计数器：提供比 1 token ≈ 4 chars 更准确的 token 估算。

支持两种后端：
- AdaptiveCounter（默认）：基于字符类型的启发式估算，零依赖
- TiktokenCounter（可选）：使用 tiktoken 精确计数，需要 pip install tiktoken
"""

import re
from abc import ABC, abstractmethod


class TokenCounter(ABC):
    """Token 计数抽象基类。"""

    @abstractmethod
    def count(self, text: str) -> int:
        """返回文本的 token 数量估算。"""
        ...

    def messages_tokens(self, messages: list) -> int:
        """估算消息列表的总 token 数。"""
        return sum(self.count(m.content) for m in messages)


class AdaptiveCounter(TokenCounter):
    """基于字符类型的自适应启发式计数器。

    比统一 1 token ≈ 4 chars 更准确：
    - CJK 字符：~0.65 token/char（中/日/韩语 1-2 token/char）
    - 英文：按单词计数 ~1.3 token/word，长词回退到 chars/4
    - 数字：~4 chars/token
    - 符号/其他：~3.5 chars/token
    """

    _CJK_RANGES = [
        (0x4E00, 0x9FFF),   # CJK Unified Ideographs
        (0x3400, 0x4DBF),   # CJK Unified Ideographs Extension A
        (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
        (0x3000, 0x303F),   # CJK Symbols and Punctuation
        (0xFF00, 0xFFEF),   # Halfwidth and Fullwidth Forms
        (0x2E80, 0x2EFF),   # CJK Radicals Supplement
        (0x31C0, 0x31EF),   # CJK Strokes
        (0xAC00, 0xD7AF),   # Hangul Syllables
        (0x3040, 0x309F),   # Hiragana
        (0x30A0, 0x30FF),   # Katakana
    ]

    _CJK_TOKEN_PER_CHAR = 0.65
    _EN_TOKEN_PER_WORD = 1.3
    _ALPHA_FLOOR_CHARS_PER_TOKEN = 4.0   # 长连续字母串的回退下限
    _DIGIT_CHARS_PER_TOKEN = 4.0
    _OTHER_CHARS_PER_TOKEN = 3.5

    def _is_cjk(self, char: str) -> bool:
        cp = ord(char)
        return any(lo <= cp <= hi for lo, hi in self._CJK_RANGES)

    def count(self, text: str) -> int:
        if not text:
            return 0

        cjk = 0
        alpha = 0
        digit = 0
        other = 0

        for ch in text:
            if self._is_cjk(ch):
                cjk += 1
            elif ch.isalpha():
                alpha += 1
            elif ch.isdigit():
                digit += 1
            elif not ch.isspace():
                other += 1

        tokens = 0.0

        # CJK: per-character
        tokens += cjk * self._CJK_TOKEN_PER_CHAR

        # Alpha: count actual words with a floor for long tokens
        if alpha > 0:
            words = len(re.findall(r'[a-zA-Z]+', text))
            word_estimate = words * self._EN_TOKEN_PER_WORD
            alpha_floor = alpha / self._ALPHA_FLOOR_CHARS_PER_TOKEN
            tokens += max(word_estimate, alpha_floor)

        # Digits: chars / ratio
        tokens += digit / self._DIGIT_CHARS_PER_TOKEN

        # Other: chars / ratio
        tokens += other / self._OTHER_CHARS_PER_TOKEN

        return max(1, round(tokens))


class TiktokenCounter(TokenCounter):
    """使用 tiktoken 进行精确计数。

    需要: pip install tiktoken
    """

    def __init__(self, model: str = "gpt-4o"):
        import tiktoken
        try:
            self._enc = tiktoken.encoding_for_model(model)
        except KeyError:
            self._enc = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        return len(self._enc.encode(text))


def create_token_counter(encoding: str = "adaptive", model: str = "gpt-4o") -> TokenCounter:
    """工厂函数：根据配置创建 token 计数器。

    Args:
        encoding: "adaptive"（默认）或 "tiktoken"
        model: 仅 tiktoken 模式使用，指定模型以匹配其 tokenizer
    """
    if encoding == "tiktoken":
        return TiktokenCounter(model=model)
    return AdaptiveCounter()
