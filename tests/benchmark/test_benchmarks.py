from agentflow.benchmark.tool_use import ToolUseBenchmark
from agentflow.benchmark.multi_hop_qa import MultiHopQABenchmark
from agentflow.benchmark.long_context import LongContextBenchmark


class TestToolUseBenchmark:
    def test_cases_defined(self):
        b = ToolUseBenchmark()
        cases = b.get_cases()
        assert len(cases) == 5

    def test_run_with_mock_agent(self):
        b = ToolUseBenchmark()
        def mock_agent(input_text: str):
            return {"tool_calls": [{"tool": "calculator", "params": {"expression": "2+2"}}]}
        report = b.run_benchmark(mock_agent)
        assert report["total"] == 5
        assert "pass_rate" in report


class TestMultiHopQA:
    def test_all_cases_need_multiple_steps(self):
        cases = MultiHopQABenchmark().get_cases()
        for c in cases:
            assert c.min_steps >= 2


class TestLongContext:
    def test_all_cases_have_long_context(self):
        cases = LongContextBenchmark().get_cases()
        for c in cases:
            assert len(c.context) >= 1000
