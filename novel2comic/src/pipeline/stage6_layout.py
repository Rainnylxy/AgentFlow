# -*- coding: utf-8 -*-
"""Stage ⑥: 漫画排版——将图片拼接成漫画页，叠加对话框和特效字。"""

import os
from PIL import Image, ImageDraw, ImageFont
from src.models import ChapterData, ComicPage
from src.llm_adapter import LLMAdapter
from src.img_adapter import ImageGenAdapter


PANEL_GAP = 20
MARGIN = 40
BUBBLE_PADDING = 12
BUBBLE_RADIUS = 16
MAX_SCROLL_WIDTH = 800


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """加载中文字体。"""
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _draw_speech_bubble(
    draw: ImageDraw.Draw,
    img_width: int,
    text: str,
    y_position: int,
    font: ImageFont.FreeTypeFont,
    style: str = "clean_rounded_rect",
):
    """在图片上绘制对话框。返回对话框占用的高度。"""
    if not text.strip():
        return 0

    max_text_width = img_width - MARGIN * 2 - BUBBLE_PADDING * 2 - 40
    lines = []
    words = list(text)
    current_line = ""
    for char in words:
        test_line = current_line + char
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] - bbox[0] > max_text_width:
            lines.append(current_line)
            current_line = char
        else:
            current_line = test_line
    if current_line:
        lines.append(current_line)

    line_height = draw.textbbox((0, 0), "啊", font=font)[3] + 4
    text_height = line_height * len(lines)
    bubble_height = text_height + BUBBLE_PADDING * 2

    bubble_x = MARGIN + 20
    bubble_w = img_width - MARGIN * 2 - 40
    bubble_y = y_position

    draw.rounded_rectangle(
        [bubble_x, bubble_y, bubble_x + bubble_w, bubble_y + bubble_height],
        radius=BUBBLE_RADIUS,
        fill=(255, 255, 255, 230),
        outline=(60, 60, 60),
        width=2,
    )

    text_y = bubble_y + BUBBLE_PADDING
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(((img_width - tw) // 2, text_y), line, fill=(20, 20, 20), font=font)
        text_y += line_height

    return bubble_height + PANEL_GAP


def _render_scroll(data: ChapterData) -> list[ComicPage]:
    """条漫模式（Webtoon/古风）：纵向拼接所有格子。"""
    pages = []

    for scene in data.scenes:
        panel_images = []
        for panel in scene.panels:
            if panel.generated_image_path and os.path.exists(panel.generated_image_path):
                panel_images.append((panel, Image.open(panel.generated_image_path)))

        if not panel_images:
            continue

        scene_width = MAX_SCROLL_WIDTH

        resized = []
        total_height = 0
        for panel, img in panel_images:
            ratio = scene_width / img.width
            new_h = int(img.height * ratio)
            img = img.resize((scene_width, new_h), Image.LANCZOS)
            resized.append((panel, img))
            total_height += new_h + PANEL_GAP

        font = _load_font(18)
        font_small = _load_font(14)
        total_height += 80 * len(resized)

        canvas = Image.new("RGB", (scene_width, total_height + MARGIN * 2), color=(30, 30, 40))
        draw = ImageDraw.Draw(canvas)

        y = MARGIN
        for panel, img in resized:
            canvas.paste(img, (0, y))
            panel_h = img.height

            if panel == resized[0][0]:
                title_font = _load_font(22)
                draw.text(
                    (20, y + 10),
                    f"场景: {scene.title}",
                    fill=(255, 255, 255),
                    font=title_font,
                )

            if panel.dialogue:
                bubble_h = _draw_speech_bubble(
                    draw, scene_width, panel.dialogue,
                    y + panel_h + 10, font,
                    data.style_profile.speech_bubble_style if data.style_profile else "clean_rounded_rect"
                )
                y += panel_h + bubble_h
            else:
                y += panel_h + PANEL_GAP

            draw.text(
                (scene_width - 80, y - 30),
                f"格{panel.panel_number}",
                fill=(150, 150, 170),
                font=font_small,
            )

        os.makedirs(os.path.join(data.output_dir, "comics"), exist_ok=True)
        output_path = os.path.join(data.output_dir, "comics", f"scene_{scene.id:02d}.png")
        canvas.save(output_path, "PNG")

        page = ComicPage(page_number=scene.id, image_path=output_path)
        pages.append(page)
        print(f"  📄 场景{scene.id} 排版完成 → {output_path}")

    return pages


def run_stage6(data: ChapterData, llm: LLMAdapter, img_gen: ImageGenAdapter) -> ChapterData:
    """⑥ 漫画排版——根据风格选择排版模式。"""

    layout_mode = data.style_profile.layout_mode if data.style_profile else "scroll"

    if layout_mode == "grid":
        print("  [INFO] Grid layout not yet implemented, falling back to scroll mode")
        data.pages = _render_scroll(data)
    else:
        data.pages = _render_scroll(data)

    print(f"\n  共生成 {len(data.pages)} 页漫画")
    return data
