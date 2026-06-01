import pytest


@pytest.fixture
def sample_workflow_definition():
    """返回一个最简 Workflow 定义，供各测试模块复用。"""
    return {
        "name": "test-workflow",
        "nodes": [
            {"id": "entry", "type": "agent", "config": {"model": "gpt-4o"}},
            {"id": "step2", "type": "agent", "config": {"model": "gpt-4o"}},
        ],
        "edges": [
            {"from": "entry", "to": "step2"},
        ],
    }
