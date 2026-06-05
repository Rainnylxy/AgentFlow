# -*- coding: utf-8 -*-
"""Pipeline 集成测试——使用 Mock LLM 验证 6 阶段端到端流程。"""

import os
import sys
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import ChapterData, AnalysisResult, CharacterAppearance, CharacterSheet
from src.llm_adapter import LLMAdapter
from src.img_adapter import ImageGenAdapter
from src.styles import STYLE_GUFENG, detect_style
from src.pipeline.engine import PipelineEngine
from src.pipeline.stage1_analyze import run_stage1
from src.pipeline.stage2_characters import run_stage2
from src.pipeline.stage3_scenes import run_stage3
from src.pipeline.stage4_storyboard import run_stage4
from src.pipeline.stage5_image_gen import run_stage5
from src.pipeline.stage6_layout import run_stage6


SAMPLE_TEXT = """
夜幕降临，长安城华灯初上。苏墨站在朱雀大街的尽头，手握一柄锈迹斑斑的铁剑。

"三年了，我终于回来了。"他低声自语，目光穿过熙攘的人潮，锁定在那座金碧辉煌的将军府上。

一个卖糖葫芦的老者经过，苏墨叫住了他："老人家，将军府近日可有什么动静？"

老者打量了他一眼，压低声音道："小兄弟，将军府三日前贴出告示，要招纳天下剑客，缉拿大盗'夜枭'。赏金一千两黄金。"

"一千两黄金..."苏墨嘴角微扬，眼中闪过一丝复杂的神色。

他绕过朱雀大街，钻进一条暗巷。一只黑猫从墙头跃下，落在他肩上。苏墨从怀中取出一张泛黄的羊皮纸，上面画着将军府的内部地形图。

"夜枭...呵，他们连我的真名都不知道了。"他收起羊皮纸，身形一闪，消失在夜色中。
"""


class MockLLM(LLMAdapter):
    """Mock LLM——返回预定义 JSON，不调用真实 API。"""

    def __init__(self):
        pass  # 不调用父类 __init__，避免需要 API key

    def chat_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.3) -> dict:
        # 根据 system prompt 内容判断是哪个阶段
        if "genre_tags" in system_prompt and "style" in system_prompt:
            # Stage 1: 分析
            return {
                "genre_tags": ["武侠", "悬疑"],
                "style": "gufeng",
                "tone": ["苍凉", "暗涌"],
                "era": "古代架空",
                "pace": "慢热",
                "characters_preview": [
                    {"name": "苏墨", "role": "主角", "first_appearance_line": "苏墨站在朱雀大街的尽头"},
                    {"name": "老者", "role": "配角", "first_appearance_line": "一个卖糖葫芦的老者"},
                    {"name": "黑猫", "role": "伙伴", "first_appearance_line": "一只黑猫从墙头跃下"},
                ],
            }
        elif "Character Sheet" in system_prompt or "sd_trigger_words" in system_prompt:
            # Stage 2: 角色设计
            return [
                {
                    "id": "su_mo",
                    "name": "苏墨",
                    "role": "protagonist",
                    "appearance": {
                        "face": "清瘦，下颌线锋利",
                        "hair": "长发束起，黑色",
                        "build": "修长精瘦",
                        "clothing": "灰色旧袍",
                        "accessories": "锈迹铁剑",
                        "distinctive_features": "锐利的眼神",
                    },
                    "sd_trigger_words": "su_mo, lean swordsman, sharp jawline, long black hair, grey robes, rusty sword",
                    "personality_notes": "冷峻内敛",
                },
                {
                    "id": "old_man",
                    "name": "老者",
                    "role": "supporting",
                    "appearance": {
                        "face": "满是皱纹",
                        "hair": "花白稀疏",
                        "build": "佝偻瘦小",
                        "clothing": "旧毡帽粗布衣",
                        "accessories": "糖葫芦小车",
                        "distinctive_features": "精明的小眼睛",
                    },
                    "sd_trigger_words": "old street vendor, weathered face, worn hat, carrying candied hawthorn sticks",
                    "personality_notes": "市井精明",
                },
                {
                    "id": "black_cat",
                    "name": "黑猫",
                    "role": "supporting",
                    "appearance": {"face": "", "hair": "", "build": "", "clothing": "", "accessories": "", "distinctive_features": "纯黑毛色"},
                    "sd_trigger_words": "black cat, sleek fur, glowing eyes, mysterious feline companion",
                    "personality_notes": "神秘伙伴",
                },
            ]
        elif "场景拆分" in system_prompt or "叙事单元" in system_prompt:
            # Stage 3: 场景拆分
            return [
                {"id": 1, "title": "朱雀大街·归来", "summary": "苏墨站在长安街头，手握锈剑锁定将军府。", "characters_in_scene": ["苏墨"], "emotion_arc": "苍凉→暗涌", "key_dialogue": "三年了，我终于回来了。"},
                {"id": 2, "title": "糖葫芦摊·情报", "summary": "苏墨向老者打探将军府消息，得知自己被以夜枭之名悬赏。", "characters_in_scene": ["苏墨", "老者"], "emotion_arc": "平静→暗讽", "key_dialogue": "赏金一千两黄金。"},
                {"id": 3, "title": "暗巷·真身", "summary": "苏墨进入暗巷，黑猫现身，他展示将军府地图。", "characters_in_scene": ["苏墨", "黑猫"], "emotion_arc": "冷静→锋芒毕露", "key_dialogue": "他们连我的真名都不知道了。"},
            ]
        elif "分镜" in system_prompt or "Panel" in system_prompt or "Storyboard" in system_prompt:
            # Stage 4: 分镜生成
            return [
                {
                    "panel_number": 1,
                    "visual_description": "远景·大俯瞰，长安城暮色四合，万家灯火，朱雀大街延伸向远方",
                    "character_action": "无人物大动作，城市运转",
                    "dialogue": "",
                    "camera_angle": "俯视大远景",
                    "mood": "繁华之下的寂寥",
                    "sd_prompt": "epic bird's eye view of ancient Chinese capital, lanterns glowing, distant mansion",
                    "character_refs": [],
                },
                {
                    "panel_number": 2,
                    "visual_description": "极近特写，一只手紧握锈迹斑斑的铁剑",
                    "character_action": "手微微收紧，指节泛白",
                    "dialogue": "",
                    "camera_angle": "极近特写",
                    "mood": "沉淀三年的沉重",
                    "sd_prompt": "extreme close-up of hand gripping rusty sword, weathered texture, melancholic",
                    "character_refs": ["苏墨"],
                },
            ]
        return {}


def test_style_detection():
    """测试风格自动判断。"""
    s = detect_style(["武侠", "悬疑"], "慢热")
    assert s.name == "gufeng", f"Expected gufeng, got {s.name}"

    s = detect_style(["校园", "恋爱"])
    assert s.name == "manga", f"Expected manga, got {s.name}"

    s = detect_style(["都市"])
    assert s.name == "webtoon", f"Expected webtoon, got {s.name}"

    print("  [PASS] test_style_detection passed")


def test_pipeline_end_to_end():
    """测试完整 Pipeline 端到端流程（Mock LLM + 占位图）。"""
    mock_llm = MockLLM()
    img_gen = ImageGenAdapter(use_placeholder=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        data = ChapterData(
            title="月下归来",
            source_text=SAMPLE_TEXT,
            output_dir=tmpdir,
        )

        # 构建 Pipeline
        engine = PipelineEngine(mock_llm, img_gen)
        engine.register(run_stage1)
        engine.register(run_stage2)
        engine.register(run_stage3)
        engine.register(run_stage4)
        engine.register(run_stage5)
        engine.register(run_stage6)

        # 运行所有阶段
        data = engine.run_all(data)

        # 验证每个阶段的输出
        assert data.current_stage == 6, f"Expected stage 6, got {data.current_stage}"
        assert data.analysis is not None, "Stage 1 should produce analysis"
        assert data.analysis.style == "gufeng", f"Expected gufeng, got {data.analysis.style}"
        assert len(data.characters) == 3, f"Expected 3 characters, got {len(data.characters)}"
        assert data.characters[0].name == "苏墨"
        assert data.characters[0].sd_trigger_words != ""
        assert len(data.scenes) == 3, f"Expected 3 scenes, got {len(data.scenes)}"
        assert len(data.scenes[0].panels) > 0, "Scene 1 should have panels"
        for s in data.scenes:
            for p in s.panels:
                assert p.status == "generated", f"Panel {p.panel_number} not generated"
                assert os.path.exists(p.generated_image_path), f"Image not found: {p.generated_image_path}"
        assert len(data.pages) > 0, "Should have at least 1 comic page"
        for page in data.pages:
            assert os.path.exists(page.image_path), f"Comic page not found: {page.image_path}"

        print("  [PASS] test_pipeline_end_to_end passed")


def test_data_serialization():
    """测试数据模型的 JSON 序列化/反序列化。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        data = ChapterData(
            title="测试",
            source_text="测试文本",
            output_dir=tmpdir,
        )
        data.analysis = AnalysisResult(genre_tags=["武侠"], style="gufeng")
        data.characters = [
            CharacterSheet(
                id="test_char", name="测试角色", role="protagonist",
                appearance=CharacterAppearance(face="测试面孔"),
                sd_trigger_words="test character trigger words",
            )
        ]

        # 保存
        filepath = os.path.join(tmpdir, "test.json")
        data.save(filepath)
        assert os.path.exists(filepath)

        # 加载
        loaded = ChapterData.load(filepath)
        assert loaded.title == "测试"
        assert loaded.analysis.style == "gufeng"
        assert len(loaded.characters) == 1
        assert loaded.characters[0].name == "测试角色"
        assert loaded.characters[0].sd_trigger_words == "test character trigger words"

        print("  [PASS] test_data_serialization passed")


if __name__ == "__main__":
    test_style_detection()
    test_pipeline_end_to_end()
    test_data_serialization()
    print("\n*** All tests passed! ***")
