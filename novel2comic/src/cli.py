# -*- coding: utf-8 -*-
"""Novel2Comic V2 CLI——交互式漫画生成工具。

用法:
    python -m src.cli "小说文本" --title "第一章"
    python -m src.cli chapter1.txt --title "月下归来"
    python -m src.cli --load projects/my_project/chapter_01.json
"""

import os
import sys
import argparse
from datetime import datetime

# 确保项目根在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import ChapterData
from src.llm_adapter import LLMAdapter
from src.img_adapter import ImageGenAdapter
from src.pipeline.engine import PipelineEngine
from src.pipeline.stage1_analyze import run_stage1
from src.pipeline.stage2_characters import run_stage2
from src.pipeline.stage3_scenes import run_stage3
from src.pipeline.stage4_storyboard import run_stage4
from src.pipeline.stage5_image_gen import run_stage5
from src.pipeline.stage6_layout import run_stage6


def _print_header(data: ChapterData):
    """打印章节信息头。"""
    print("\n" + "=" * 60)
    print(f"  📖 {data.title}")
    print(f"  文本长度: {len(data.source_text)} 字符")
    print(f"  风格: {data.style_profile.name if data.style_profile else 'auto'}")
    print("=" * 60)


def _print_status(data: ChapterData, engine: PipelineEngine):
    """打印当前进度。"""
    print(f"\n  当前进度: {engine.stage_name(data.current_stage)}")
    if data.current_stage < engine.total_stages:
        print(f"  下一步:   {engine.stage_name(data.current_stage + 1)}")
    print(f"  角色: {len(data.characters)} 个")
    print(f"  场景: {len(data.scenes)} 个")
    panels = sum(len(s.panels) for s in data.scenes)
    print(f"  分镜: {panels} 格")


def _show_menu() -> str:
    """显示交互菜单。"""
    print("\n" + "-" * 40)
    print("  [c] 继续下一阶段")
    print("  [r] 重做当前阶段")
    print("  [v] 查看当前数据摘要")
    print("  [s] 保存进度")
    print("  [q] 保存并退出")
    print("-" * 40)
    return input("  > ").strip().lower()


def _show_summary(data: ChapterData):
    """简要展示当前数据。"""
    print("\n  ── 数据摘要 ──")
    if data.analysis:
        print(f"  风格: {data.analysis.style} | 基调: {data.analysis.tone}")
    print(f"  角色 ({len(data.characters)}):")
    for c in data.characters:
        print(f"    - {c.name} [{c.role}] {c.sd_trigger_words[:50]}...")
    print(f"  场景 ({len(data.scenes)}):")
    for s in data.scenes:
        print(f"    - 场景{s.id}: {s.title} ({len(s.panels)}格)")
        for p in s.panels:
            status_icon = "✅" if p.status == "generated" else "⏳"
            print(f"      {status_icon} 格{p.panel_number}: {p.visual_description[:40]}...")


def main():
    parser = argparse.ArgumentParser(description="Novel2Comic V2 - 小说转漫画")
    parser.add_argument("input", nargs="?", help="小说文本或文件路径")
    parser.add_argument("--title", "-t", default="未命名章节", help="章节标题")
    parser.add_argument("--load", "-l", help="从 JSON 文件恢复进度")
    parser.add_argument("--auto", "-a", action="store_true", help="全自动模式（无交互）")
    args = parser.parse_args()

    # 初始化 LLM 和 ImageGen
    try:
        llm = LLMAdapter()
    except Exception as e:
        print(f"[!] LLM 初始化失败: {e}")
        print("[!] 请设置 N2C_LLM_API_KEY 环境变量")
        sys.exit(1)

    img_gen = ImageGenAdapter()

    # 加载或新建数据
    if args.load:
        data = ChapterData.load(args.load)
        print(f"[+] 从 {args.load} 恢复进度")
    else:
        text = args.input or ""
        if not text:
            parser.print_help()
            sys.exit(1)
        if os.path.isfile(text):
            with open(text, "r", encoding="utf-8") as f:
                text = f.read()
            if args.title == "未命名章节":
                args.title = os.path.splitext(os.path.basename(text))[0]

        # 初始化数据
        project_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "projects",
            datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
        os.makedirs(project_dir, exist_ok=True)

        data = ChapterData(
            title=args.title,
            source_text=text,
            output_dir=project_dir,
            created_at=datetime.now().isoformat(),
        )

    # 构建 Pipeline
    engine = PipelineEngine(llm, img_gen)
    engine.register(run_stage1)
    engine.register(run_stage2)
    engine.register(run_stage3)
    engine.register(run_stage4)
    engine.register(run_stage5)
    engine.register(run_stage6)

    # 主循环
    while data.current_stage < engine.total_stages:
        _print_header(data)
        _print_status(data, engine)

        if args.auto:
            data = engine.run_stage(data, data.current_stage)
            continue

        choice = _show_menu()

        if choice == "c":
            data = engine.run_stage(data, data.current_stage)
            save_path = os.path.join(data.output_dir, "chapter_data.json")
            data.save(save_path)
            print(f"  [SAVE] 已自动保存到 {save_path}")

        elif choice == "r":
            prev_stage = max(0, data.current_stage - 1)
            if data.current_stage == 1:
                data.analysis = None
            elif data.current_stage == 2:
                data.characters = []
            elif data.current_stage == 3:
                data.scenes = []
            elif data.current_stage == 4:
                for s in data.scenes:
                    s.panels = []
            elif data.current_stage == 5:
                for s in data.scenes:
                    for p in s.panels:
                        p.generated_image_path = ""
                        p.status = "pending"
            elif data.current_stage == 6:
                data.pages = []
            data.current_stage = prev_stage
            print(f"  [BACK] 已回退到 {engine.stage_name(data.current_stage)}")

        elif choice == "v":
            _show_summary(data)

        elif choice == "s":
            save_path = os.path.join(data.output_dir, "chapter_data.json")
            data.save(save_path)
            print(f"  [SAVE] 已保存到 {save_path}")

        elif choice == "q":
            save_path = os.path.join(data.output_dir, "chapter_data.json")
            data.save(save_path)
            print(f"  [SAVE] 已保存到 {save_path}")
            print("  [BYE] 再见！")
            break

    if data.current_stage >= engine.total_stages:
        print("\n" + "=" * 60)
        print("  [DONE] 全部 6 阶段完成！")
        print(f"  输出目录: {data.output_dir}")
        for page in data.pages:
            print(f"  [PAGE] {page.image_path}")
        print("=" * 60)


if __name__ == "__main__":
    main()
