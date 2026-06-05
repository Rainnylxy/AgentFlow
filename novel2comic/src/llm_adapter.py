# -*- coding: utf-8 -*-
"""LLM Adapter——封装 LLM 调用。"""

import os
import json
from openai import OpenAI


class LLMAdapter:
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
        full_system = (
            system_prompt
            + "\n\nYou MUST respond with valid JSON only. No markdown fences, no explanation."
        )
        text = self.chat(full_system, user_prompt, temperature)
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        return json.loads(text)
