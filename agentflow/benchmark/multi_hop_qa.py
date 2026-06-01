"""Multi-Hop QA Benchmark"""

from dataclasses import dataclass


@dataclass
class MultiHopCase:
    id: str
    input: str
    expected_answer: str
    min_steps: int
    required_tools: list[str]


class MultiHopQABenchmark:
    def get_cases(self) -> list[MultiHopCase]:
        return [
            MultiHopCase("mh1", "Who is the CEO of the company that made ChatGPT?",
                         "Sam Altman", 2, ["search"]),
            MultiHopCase("mh2", "What is the population of the capital of France?",
                         "~2.1 million", 2, ["search"]),
            MultiHopCase("mh3", "Which has more GitHub stars: LangChain or AutoGen?",
                         "LangChain", 3, ["github_search"]),
        ]
