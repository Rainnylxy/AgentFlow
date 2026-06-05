# -*- coding: utf-8 -*-
"""Pipeline 引擎——编排 6 个阶段顺序执行，每阶段间可暂停。"""

from typing import Callable
from src.models import ChapterData
from src.llm_adapter import LLMAdapter
from src.img_adapter import ImageGenAdapter


# 阶段函数签名
StageFn = Callable[[ChapterData, LLMAdapter, ImageGenAdapter], ChapterData]


class PipelineEngine:
    """Pipeline 编排器。

    负责按顺序调用 6 个阶段函数，管理当前进度，支持单步执行和完整运行。
    """

    STAGE_NAMES = {
        0: "未开始",
        1: "① 文本分析",
        2: "② 角色设计",
        3: "③ 场景拆分",
        4: "④ 分镜生成",
        5: "⑤ 图像生成",
        6: "⑥ 漫画排版",
    }

    def __init__(self, llm: LLMAdapter, img_gen: ImageGenAdapter):
        self._stages: list[StageFn] = []
        self.llm = llm
        self.img_gen = img_gen

    def register(self, stage_fn: StageFn):
        """注册一个阶段函数。按注册顺序执行。"""
        self._stages.append(stage_fn)

    def run_stage(self, data: ChapterData, stage_index: int) -> ChapterData:
        """执行单个阶段。stage_index 从 0 开始。"""
        if stage_index >= len(self._stages):
            raise ValueError(f"Stage {stage_index} not registered (total: {len(self._stages)})")

        print(f"\n{'='*50}")
        print(f"  {self.STAGE_NAMES.get(stage_index + 1, f'Stage {stage_index+1}')}")
        print(f"{'='*50}")

        stage_fn = self._stages[stage_index]
        data = stage_fn(data, self.llm, self.img_gen)
        data.current_stage = stage_index + 1

        print(f"  [OK] 完成")
        return data

    def run_all(self, data: ChapterData) -> ChapterData:
        """完整运行所有已注册的阶段（无暂停）。"""
        for i in range(len(self._stages)):
            data = self.run_stage(data, i)
        return data

    @property
    def total_stages(self) -> int:
        return len(self._stages)

    def stage_name(self, index: int) -> str:
        return self.STAGE_NAMES.get(index + 1, f"Stage {index+1}")
