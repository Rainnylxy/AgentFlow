"""Tool-Use Benchmark"""

from dataclasses import dataclass
from typing import Callable


@dataclass
class ToolUseCase:
    id: str
    input: str
    expected_tool: str
    expected_params: dict


class ToolUseBenchmark:
    def get_cases(self) -> list[ToolUseCase]:
        return [
            ToolUseCase("t1", "Calculate 2+2", "calculator", {"expression": "2+2"}),
            ToolUseCase("t2", "Weather in Beijing?", "weather", {"city": "Beijing"}),
            ToolUseCase("t3", "Search AgentFlow on GitHub", "github_search", {"query": "AgentFlow"}),
            ToolUseCase("t4", "Translate 'hello' to Chinese", "translate", {"text": "hello", "target": "zh"}),
            ToolUseCase("t5", "Calculate 100*50", "calculator", {"expression": "100*50"}),
        ]

    def run_benchmark(self, agent_fn: Callable) -> dict:
        cases = self.get_cases()
        results = []
        for case in cases:
            output = agent_fn(case.input)
            tool_calls = output.get("tool_calls", [])
            passed = any(tc["tool"] == case.expected_tool for tc in tool_calls)
            results.append({"case_id": case.id, "passed": passed})
        passed = sum(1 for r in results if r["passed"])
        return {"total": len(results), "passed": passed,
                "pass_rate": passed / len(results) if results else 0, "details": results}
