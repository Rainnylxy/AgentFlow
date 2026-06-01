"""Long-Context Benchmark"""

from dataclasses import dataclass


@dataclass
class LongContextCase:
    id: str
    input: str
    context: str
    expected: str


class LongContextBenchmark:
    def get_cases(self) -> list[LongContextCase]:
        base = "AgentFlow is a production-grade multi-agent orchestration framework. "
        return [
            LongContextCase("lc1", "What is AgentFlow?", base * 50, "orchestration framework"),
            LongContextCase("lc2", "List three features of AgentFlow",
                            base * 30 + "Features: DAG execution, circuit breaker, eval engine. " * 10,
                            "DAG"),
        ]
