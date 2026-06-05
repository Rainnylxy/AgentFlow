# -*- coding: utf-8 -*-
"""ImageGen Adapter——封装云端生图 API + 本地占位图兜底。"""

import os
import io
import uuid
from PIL import Image, ImageDraw, ImageFont


class ImageGenAdapter:
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
        img = Image.new("RGB", (width, height), color=(40, 40, 50))
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, width - 1, height - 1], outline=(100, 100, 120), width=3)
        summary = prompt[:80] + ("..." if len(prompt) > 80 else "")
        lines = [summary[i:i+40] for i in range(0, len(summary), 40)]
        font = None
        for font_path in [
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simsun.ttc",
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
        from openai import OpenAI
        import httpx
        import requests as req

        proxy = os.getenv("N2C_PROXY", "")
        http_client = httpx.Client(proxy=proxy) if proxy else None
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url or "https://api.openai.com/v1",
            http_client=http_client,
        )
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        img_response = req.get(image_url)
        img = Image.open(io.BytesIO(img_response.content))
        img = img.resize((width, height), Image.LANCZOS)
        img.save(filepath, "PNG")
