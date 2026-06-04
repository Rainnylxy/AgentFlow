# -*- coding: utf-8 -*-
"""
Novel2Comic 示例脚本

用法:
    export AGENTFLOW_API_KEY='sk-your-key'
    python example.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from novel2comic.agent import run_novel2comic


SAMPLE_TEXT = """
夜幕降临，长安城华灯初上。苏墨站在朱雀大街的尽头，手握一柄锈迹斑斑的铁剑。

"三年了，我终于回来了。"他低声自语，目光穿过熙攘的人潮，锁定在那座金碧辉煌的将军府上。

一个卖糖葫芦的老者经过，苏墨叫住了他："老人家，将军府近日可有什么动静？"

老者打量了他一眼，压低声音道："小兄弟，你怕是外地来的吧？将军府三日前贴出告示，要招纳天下剑客，说是要缉拿一个叫'夜枭'的大盗。赏金一千两黄金。"

"一千两黄金..."苏墨嘴角微扬，眼中闪过一丝复杂的神色。

他绕过朱雀大街，钻进一条暗巷。一只黑猫从墙头跃下，落在他肩上。苏墨从怀中取出一张泛黄的羊皮纸，上面画着将军府的内部地形图。

"夜枭...呵，他们连我的真名都不知道了。"他收起羊皮纸，身形一闪，消失在夜色中。
"""


async def main():
    print("Novel2Comic Agent 示例")
    print(f"输入文本: {len(SAMPLE_TEXT)} 字符\n")

    result = await run_novel2comic(SAMPLE_TEXT, "月下归来")
    print("\n" + "=" * 60)
    print("生成的分镜脚本:")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
