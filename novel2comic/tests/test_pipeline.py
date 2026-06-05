# -*- coding: utf-8 -*-
"""Agent 集成测试——使用 Mock LLM 验证 Agent Tool 端到端流程。"""

import os
import sys
import json
import tempfile
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 也需要项目根来 import novel2comic 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models import ChapterData, AnalysisResult, CharacterAppearance, CharacterSheet
from src.img_adapter import ImageGenAdapter
from src.styles import detect_style


SAMPLE_TEXT = """
夜幕降临，长安城华灯初上。苏墨站在朱雀大街的尽头，手握一柄锈迹斑斑的铁剑。

"三年了，我终于回来了。"他低声自语，目光穿过熙攘的人潮，锁定在那座金碧辉煌的将军府上。

一个卖糖葫芦的老者经过，苏墨叫住了他："老人家，将军府近日可有什么动静？"

老者打量了他一眼，压低声音道："小兄弟，将军府三日前贴出告示，要招纳天下剑客，缉拿大盗'夜枭'。赏金一千两黄金。"

"一千两黄金..."苏墨嘴角微扬，眼中闪过一丝复杂的神色。

他绕过朱雀大街，钻进一条暗巷。一只黑猫从墙头跃下，落在他肩上。苏墨从怀中取出一张泛黄的羊皮纸，上面画着将军府的内部地形图。

"夜枭...呵，他们连我的真名都不知道了。"他收起羊皮纸，身形一闪，消失在夜色中。
"""


# ============================================================
# Mock: 模拟 OpenAI 同步客户端
# ============================================================

class MockCompletion:
    def __init__(self, content):
        self.choices = [MagicMock()]
        self.choices[0].message.content = content


class MockChat:
    def __init__(self, responses: list[dict]):
        self.responses = responses
        self.call_count = 0

    def create(self, **kwargs):
        idx = self.call_count
        self.call_count += 1
        if idx < len(self.responses):
            return MockCompletion(json.dumps(self.responses[idx], ensure_ascii=False))
        return MockCompletion("{}")


class MockOpenAI:
    def __init__(self, responses: list[dict]):
        self.chat = MagicMock()
        self.chat.completions = MockChat(responses)


def test_style_detection():
    """测试风格自动判断。"""
    s = detect_style(["武侠", "悬疑"], "慢热")
    assert s.name == "gufeng", f"Expected gufeng, got {s.name}"

    s = detect_style(["校园", "恋爱"])
    assert s.name == "manga", f"Expected manga, got {s.name}"

    s = detect_style(["都市"])
    assert s.name == "webtoon", f"Expected webtoon, got {s.name}"

    print("  [PASS] test_style_detection passed")


def test_agent_tools_end_to_end():
    """测试 Agent Tool 端到端流程（Mock LLM）。"""
    import novel2comic.agent as agent_module

    mock_responses = [
        # Stage 1: analyze_text 返回
        {
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
        },
        # Stage 2: design_characters 返回
        [
            {
                "id": "su_mo", "name": "苏墨", "role": "protagonist",
                "appearance": {"face": "清瘦", "hair": "长发", "build": "修长", "clothing": "灰袍", "accessories": "锈剑", "distinctive_features": "锐利眼神"},
                "sd_trigger_words": "su_mo, lean swordsman, sharp jawline, grey robes, rusty sword",
                "personality_notes": "冷峻内敛",
            },
            {
                "id": "old_man", "name": "老者", "role": "supporting",
                "appearance": {"face": "皱纹", "hair": "花白", "build": "佝偻", "clothing": "粗布衣", "accessories": "糖葫芦车", "distinctive_features": "精明小眼"},
                "sd_trigger_words": "old street vendor, weathered face, worn hat",
                "personality_notes": "市井精明",
            },
            {
                "id": "black_cat", "name": "黑猫", "role": "supporting",
                "appearance": {"face": "", "hair": "", "build": "", "clothing": "", "accessories": "", "distinctive_features": "纯黑毛色"},
                "sd_trigger_words": "black cat, sleek fur, glowing eyes",
                "personality_notes": "神秘伙伴",
            },
        ],
        # Stage 3: extract_scenes 返回
        [
            {"id": 1, "title": "朱雀大街·归来", "summary": "苏墨归来", "characters_in_scene": ["苏墨"], "emotion_arc": "苍凉→暗涌", "key_dialogue": "三年了"},
            {"id": 2, "title": "糖葫芦摊·情报", "summary": "打探消息", "characters_in_scene": ["苏墨", "老者"], "emotion_arc": "平静→暗讽", "key_dialogue": "一千两黄金"},
        ],
        # Stage 4: storyboard_scene scene 1
        [
            {"panel_number": 1, "visual_description": "远景长安", "character_action": "无", "dialogue": "", "camera_angle": "俯视大远景", "mood": "寂寥", "sd_prompt": "epic view of capital", "character_refs": []},
            {"panel_number": 2, "visual_description": "锈剑特写", "character_action": "手握紧", "dialogue": "", "camera_angle": "极近特写", "mood": "沉重", "sd_prompt": "close-up rusty sword", "character_refs": ["苏墨"]},
        ],
        # Stage 4: storyboard_scene scene 2
        [
            {"panel_number": 1, "visual_description": "街边对话", "character_action": "苏墨拦下老者", "dialogue": "将军府近日可有什么动静？", "camera_angle": "中景", "mood": "试探", "sd_prompt": "street conversation", "character_refs": ["苏墨", "老者"]},
            {"panel_number": 2, "visual_description": "老者密语", "character_action": "压低声音", "dialogue": "赏金一千两黄金", "camera_angle": "近景", "mood": "暗讽", "sd_prompt": "old man whispering", "character_refs": ["老者"]},
        ],
    ]

    mock_client = MockOpenAI(mock_responses)
    img_gen = ImageGenAdapter(use_placeholder=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        # 注入 Agent 上下文
        agent_module._ctx.data = ChapterData(
            title="月下归来",
            source_text=SAMPLE_TEXT,
            output_dir=tmpdir,
        )
        agent_module._ctx.openai_client = mock_client
        agent_module._ctx.llm_model = "mock"
        agent_module._ctx.img_gen = img_gen

        # === 逐个调用 Tool ===

        # Tool 1: analyze_text
        result1 = json.loads(agent_module.analyze_text.func(SAMPLE_TEXT))
        assert result1["status"] == "ok", f"analyze_text failed: {result1}"
        assert agent_module._ctx.data.analysis is not None
        assert agent_module._ctx.data.analysis.style == "gufeng"
        assert len(agent_module._ctx.data.analysis.characters_preview) == 3
        print("  [PASS] Tool 1: analyze_text")

        # Tool 2: design_characters
        result2 = json.loads(agent_module.design_characters.func())
        assert result2["status"] == "ok", f"design_characters failed: {result2}"
        assert len(agent_module._ctx.data.characters) == 3
        assert agent_module._ctx.data.characters[0].name == "苏墨"
        assert agent_module._ctx.data.characters[0].sd_trigger_words != ""
        print("  [PASS] Tool 2: design_characters")

        # Tool 3: extract_scenes
        result3 = json.loads(agent_module.extract_scenes.func())
        assert result3["status"] == "ok", f"extract_scenes failed: {result3}"
        assert len(agent_module._ctx.data.scenes) == 2
        assert agent_module._ctx.data.scenes[0].title == "朱雀大街·归来"
        print("  [PASS] Tool 3: extract_scenes")

        # Tool 4: storyboard_scene (scene 1)
        result4a = json.loads(agent_module.storyboard_scene.func(1))
        assert result4a["status"] == "ok", f"storyboard_scene(1) failed: {result4a}"
        assert len(agent_module._ctx.data.scenes[0].panels) == 2
        print("  [PASS] Tool 4a: storyboard_scene(scene_id=1)")

        # Tool 4: storyboard_scene (scene 2)
        result4b = json.loads(agent_module.storyboard_scene.func(2))
        assert result4b["status"] == "ok", f"storyboard_scene(2) failed: {result4b}"
        assert len(agent_module._ctx.data.scenes[1].panels) == 2
        print("  [PASS] Tool 4b: storyboard_scene(scene_id=2)")

        # Verify sd_prompt was enhanced with style base + character triggers + aspect ratio
        panel1_prompt = agent_module._ctx.data.scenes[1].panels[0].sd_prompt
        assert "webtoon" in panel1_prompt.lower() or "gufeng" in panel1_prompt.lower() or "manga" in panel1_prompt.lower(), \
            f"sd_prompt should contain style base: {panel1_prompt[:100]}"
        print("  [PASS] sd_prompt auto-enhancement verified")

        # Tool 5: generate_images
        result5 = json.loads(agent_module.generate_images.func(0))
        assert result5["status"] == "ok", f"generate_images failed: {result5}"
        assert result5["generated"] == 4  # 2 scenes x 2 panels each
        for scene in agent_module._ctx.data.scenes:
            for panel in scene.panels:
                assert panel.status == "generated"
                assert os.path.exists(panel.generated_image_path)
        print("  [PASS] Tool 5: generate_images")

        # Tool 6: compile_comic
        result6 = json.loads(agent_module.compile_comic.func())
        assert result6["status"] == "ok", f"compile_comic failed: {result6}"
        assert result6["page_count"] == 2
        for page in agent_module._ctx.data.pages:
            assert os.path.exists(page.image_path)
        print("  [PASS] Tool 6: compile_comic")

        # Tool 7: save_project
        result7 = json.loads(agent_module.save_project.func())
        assert result7["status"] == "ok", f"save_project failed: {result7}"
        assert os.path.exists(result7["path"])
        print("  [PASS] Tool 7: save_project")

        # Verify save/load roundtrip
        loaded = ChapterData.load(result7["path"])
        assert loaded.title == "月下归来"
        assert len(loaded.characters) == 3
        assert len(loaded.scenes) == 2
        print("  [PASS] Save/load roundtrip")

    print("  [PASS] test_agent_tools_end_to_end passed")


def test_data_serialization():
    """测试数据模型 JSON 序列化。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        data = ChapterData(title="测试", source_text="测试文本", output_dir=tmpdir)
        data.analysis = AnalysisResult(genre_tags=["武侠"], style="gufeng")
        data.characters = [
            CharacterSheet(
                id="test_char", name="测试角色", role="protagonist",
                appearance=CharacterAppearance(face="测试面孔"),
                sd_trigger_words="test character trigger words",
            )
        ]
        filepath = os.path.join(tmpdir, "test.json")
        data.save(filepath)
        loaded = ChapterData.load(filepath)
        assert loaded.title == "测试"
        assert loaded.analysis.style == "gufeng"
        assert len(loaded.characters) == 1
        assert loaded.characters[0].name == "测试角色"
        print("  [PASS] test_data_serialization passed")


if __name__ == "__main__":
    test_style_detection()
    test_agent_tools_end_to_end()
    test_data_serialization()
    print("\n*** All tests passed! ***")
