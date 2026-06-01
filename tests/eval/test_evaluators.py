import pytest
from agentflow.eval.exact_match import ExactMatchEvaluator
from agentflow.eval.trajectory import TrajectoryEvaluator
from agentflow.eval.suite import EvalSuite, EvalCase


class TestExactMatch:
    def test_perfect_match(self):
        r = ExactMatchEvaluator().evaluate("hello", "hello")
        assert r.score == 1.0 and r.passed

    def test_mismatch(self):
        r = ExactMatchEvaluator().evaluate("hello", "world")
        assert r.score == 0.0 and not r.passed

    def test_case_insensitive(self):
        r = ExactMatchEvaluator(case_sensitive=False).evaluate("Hello", "hello")
        assert r.score == 1.0

    def test_json_match(self):
        r = ExactMatchEvaluator().evaluate('{"a":1,"b":2}', '{"b":2,"a":1}')
        assert r.score == 1.0 and r.passed

    def test_whitespace_normalization(self):
        r = ExactMatchEvaluator(normalize_whitespace=True).evaluate("a b", "  a   b  ")
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


class TestEvalSuite:
    def test_run_suite(self):
        e = ExactMatchEvaluator()
        suite = EvalSuite("test", [
            EvalCase("c1", "Hi", "Hi", e),
            EvalCase("c2", "Hi", "Bye", e),
        ])
        report = suite.run(lambda x: "Hi")
        assert report.passed == 1 and report.failed == 1 and report.pass_rate == 0.5

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="No cases"):
            EvalSuite("empty", []).run(lambda x: x)

    def test_compare(self):
        e = ExactMatchEvaluator()
        suite = EvalSuite("cmp", [EvalCase("c", "Hi", "Hi", e)])
        old = suite.run(lambda x: "Hi")
        new = suite.run(lambda x: "Hi")
        diff = suite.compare(old, new)
        assert diff["unchanged"] == 1
