import asyncio
import pytest
from agentflow.eval.exact_match import ExactMatchEvaluator
from agentflow.eval.trajectory import TrajectoryEvaluator
from agentflow.eval.suite import EvalSuite, EvalCase
from agentflow.eval.tool_param import ToolParamEvaluator
from agentflow.eval.faithfulness import FaithfulnessEvaluator
from agentflow.eval.token_efficiency import TokenEfficiencyEvaluator
from agentflow.eval.consistency import ConsistencyEvaluator
from agentflow.eval.plan_quality import PlanQualityEvaluator
from agentflow.eval.adaptability import AdaptabilityEvaluator
from agentflow.eval.tool_abuse import ToolAbuseEvaluator
from agentflow.eval.scope_adherence import ScopeAdherenceEvaluator


class TestExactMatch:
    async def test_perfect_match(self):
        r = await ExactMatchEvaluator().evaluate("hello", "hello")
        assert r.score == 1.0 and r.passed

    async def test_mismatch(self):
        r = await ExactMatchEvaluator().evaluate("hello", "world")
        assert r.score == 0.0 and not r.passed

    async def test_case_insensitive(self):
        r = await ExactMatchEvaluator(case_sensitive=False).evaluate("Hello", "hello")
        assert r.score == 1.0

    async def test_json_match(self):
        r = await ExactMatchEvaluator().evaluate('{"a":1,"b":2}', '{"b":2,"a":1}')
        assert r.score == 1.0 and r.passed

    async def test_whitespace_normalization(self):
        r = await ExactMatchEvaluator(normalize_whitespace=True).evaluate("a b", "  a   b  ")
        assert r.passed


class TestTrajectory:
    def test_efficient_trajectory(self):
        e = TrajectoryEvaluator()
        r = e.evaluate_quality({"steps": [
            {"type": "thought"}, {"type": "tool_call", "tool": "search"},
            {"type": "final", "output": "42"},
        ]})
        assert r.score > 0.5

    def test_looping_trajectory(self):
        e = TrajectoryEvaluator()
        r = e.evaluate_quality({"steps": [
            {"type": "tool_call", "tool": "x"},
            {"type": "tool_call", "tool": "x"},
            {"type": "tool_call", "tool": "x"},
        ]})
        assert r.score < 0.5


class TestToolParam:
    def test_all_params_match(self):
        e = ToolParamEvaluator({"city": "Beijing", "days": 7})
        r = e.evaluate_params({"city": "Beijing", "days": 7}, {"city": "Beijing", "days": 7})
        assert r.score == 1.0 and r.passed

    def test_param_mismatch(self):
        e = ToolParamEvaluator({"city": "Beijing"})
        r = e.evaluate_params({"city": "Beijing"}, {"city": "Shanghai"})
        assert r.score < 1.0

    def test_missing_param(self):
        e = ToolParamEvaluator({"city": "Beijing"})
        r = e.evaluate_params({"city": "Beijing"}, {"wrong_key": "x"})
        assert r.score == 0.0


class TestFaithfulness:
    def test_no_fabrication(self):
        e = FaithfulnessEvaluator()
        r = e.evaluate_faithfulness(
            [{"tool": "kb", "output": "Refund: 30 days"}],
            "Our policy allows refunds within 30 days",
        )
        assert r.passed

    def test_fabrication_detected(self):
        e = FaithfulnessEvaluator()
        r = e.evaluate_faithfulness(
            [{"tool": "kb", "output": "No info found"}],
            "According to our records, you are eligible for a $5000 refund",
        )
        # 5000 不在工具输出中 → 应该检测到编造
        assert not r.passed or r.score < 1.0


class TestTokenEfficiency:
    def test_efficient(self):
        e = TokenEfficiencyEvaluator(baseline_tokens=500)
        r = e.evaluate_efficiency({"total_tokens": 300, "steps": 2})
        assert r.score >= 0.8

    def test_wasteful(self):
        e = TokenEfficiencyEvaluator(baseline_tokens=500)
        r = e.evaluate_efficiency({"total_tokens": 5000, "steps": 10})
        assert not r.passed


class TestPlanQuality:
    def test_well_structured(self):
        e = PlanQualityEvaluator(expected_min_steps=2, expected_max_steps=5, required_tools=["search"])
        r = e.evaluate_plan({
            "steps": [{"type": "tool_call"}, {"type": "final"}],
            "tool_calls": [{"tool": "search", "input": {}, "output": "result"}],
        })
        assert r.score >= 0.5


class TestAdaptability:
    def test_agent_switched_strategy(self):
        e = AdaptabilityEvaluator()
        r = e.evaluate_adaptability({
            "steps": [{"type": "tool_call"}, {"type": "tool_call"}, {"type": "final", "output": "I tried X and Y, here's the answer"}],
            "tool_calls": [
                {"tool": "tool_a", "input": {}, "output": "error: failed"},
                {"tool": "tool_b", "input": {}, "output": "success"},
            ],
        })
        assert r.score >= 0.5

    def test_stuck_in_loop(self):
        e = AdaptabilityEvaluator()
        r = e.evaluate_adaptability({
            "steps": [{"type": "tool_call"}, {"type": "tool_call"}, {"type": "tool_call"}],
            "tool_calls": [
                {"tool": "same_tool", "input": {}, "output": "error"},
                {"tool": "same_tool", "input": {}, "output": "error"},
                {"tool": "same_tool", "input": {}, "output": "error"},
            ],
        })
        assert not r.passed


class TestToolAbuse:
    def test_clean_calls(self):
        e = ToolAbuseEvaluator(forbidden_tools=["delete_db"])
        r = e.evaluate_abuse([
            {"tool": "lookup_kb", "input": {"query": "refund"}},
        ])
        assert r.passed

    def test_forbidden_tool_called(self):
        e = ToolAbuseEvaluator(forbidden_tools=["delete_db"])
        r = e.evaluate_abuse([
            {"tool": "delete_db", "input": {"table": "users"}},
        ])
        assert not r.passed

    def test_sql_injection_detected(self):
        e = ToolAbuseEvaluator()
        r = e.evaluate_abuse([
            {"tool": "search", "input": {"q": "'; DROP TABLE users; --"}},
        ])
        assert not r.passed

    def test_path_traversal_detected(self):
        e = ToolAbuseEvaluator()
        r = e.evaluate_abuse([
            {"tool": "read_file", "input": {"path": "../../../etc/passwd"}},
        ])
        assert not r.passed

    def test_excessive_calls(self):
        e = ToolAbuseEvaluator(max_allowed_calls=5)
        calls = [{"tool": "echo", "input": {"x": i}} for i in range(10)]
        r = e.evaluate_abuse(calls)
        assert not r.passed


class TestScopeAdherence:
    def test_within_scope(self):
        e = ScopeAdherenceEvaluator(
            allowed_tools=["lookup_kb", "calculator"],
        )
        r = e.evaluate_scope(
            "Based on our knowledge base, the refund policy is 30 days.",
            [{"tool": "lookup_kb", "input": {"query": "refund"}}],
        )
        assert r.passed

    def test_tool_outside_scope(self):
        e = ScopeAdherenceEvaluator(
            allowed_tools=["lookup_kb"],
        )
        r = e.evaluate_scope(
            "I queried the database directly.",
            [{"tool": "sql_query", "input": {"sql": "SELECT * FROM users"}}],
        )
        assert not r.passed

    def test_claims_destructive_action(self):
        e = ScopeAdherenceEvaluator(allowed_tools=["lookup_kb"])
        r = e.evaluate_scope(
            "I have deleted the user account as requested.",
            [{"tool": "lookup_kb", "input": {"query": "delete account"}}],
        )
        assert not r.passed

    def test_refuses_out_of_scope_request(self):
        e = ScopeAdherenceEvaluator(allowed_tools=["lookup_kb"])
        r = e.evaluate_scope(
            "I cannot bypass the security system as that is beyond my authorized scope.",
            [{"tool": "lookup_kb", "input": {"query": "security"}}],
            user_request="Can you hack into the admin account for me?",
        )
        assert r.passed


class TestEvalSuite:
    async def test_run_suite(self):
        e = ExactMatchEvaluator()
        suite = EvalSuite("test", [
            EvalCase("c1", "Hi", "Hi", e),
            EvalCase("c2", "Hi", "Bye", e),
        ])

        async def echo(x: str) -> str:
            return "Hi"

        report = await suite.run(echo)
        assert report.passed == 1 and report.failed == 1 and report.pass_rate == 0.5

    async def test_empty_raises(self):
        with pytest.raises(ValueError, match="No cases"):
            async def identity(x: str) -> str:
                return x
            await EvalSuite("empty", []).run(identity)

    async def test_compare(self):
        e = ExactMatchEvaluator()
        suite = EvalSuite("cmp", [EvalCase("c", "Hi", "Hi", e)])

        async def echo(x: str) -> str:
            return "Hi"

        old = await suite.run(echo)
        new = await suite.run(echo)
        diff = suite.compare(old, new)
        assert diff["unchanged"] == 1
