import pytest
from agentflow.runtime.thinking.adaptive import AdaptiveRouter
from agentflow.runtime.thinking.react import ReActStrategy
from agentflow.runtime.thinking.plan_execute import PlanExecuteStrategy
from agentflow.runtime.thinking.cot import CoTStrategy
from agentflow.runtime.thinking.reflection import ReflectionWrapper


class TestAdaptiveRouter:
    def test_routes_simple_to_react(self):
        router = AdaptiveRouter()
        strategy = router.route("What is the weather in Beijing?", [])
        assert isinstance(strategy, ReActStrategy)

    def test_routes_multi_step_to_plan_execute(self):
        router = AdaptiveRouter()
        strategy = router.route(
            "First check the weather, then book a hotel, then send me a confirmation", []
        )
        assert isinstance(strategy, PlanExecuteStrategy)

    def test_routes_reasoning_to_cot(self):
        router = AdaptiveRouter()
        strategy = router.route("Prove that the sum of angles in a triangle is 180 degrees", [])
        assert isinstance(strategy, CoTStrategy)

    def test_routes_safe_critical_to_reflection(self):
        router = AdaptiveRouter()
        strategy = router.route("Delete the production database and redeploy", [])
        assert isinstance(strategy, ReflectionWrapper)

    def test_default_to_react(self):
        router = AdaptiveRouter()
        strategy = router.route("Hello!", [])
        assert isinstance(strategy, ReActStrategy)

    def test_multi_step_with_safe_triggers_reflection(self):
        """多步 + 高风险 → PlanExecute wrapped in Reflection。"""
        router = AdaptiveRouter()
        strategy = router.route(
            "First delete the old config, then deploy the new version, then verify",
            []
        )
        assert isinstance(strategy, ReflectionWrapper)
