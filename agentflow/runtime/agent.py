"""Agent 基类"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from agentflow.runtime.llm_client import LLMClient
from agentflow.runtime.tool_registry import ToolRegistry
from agentflow.runtime.memory import MemoryManager


@dataclass
class AgentResult:
    output: str
    tool_calls: list = field(default_factory=list)
    steps: list = field(default_factory=list)


class BaseAgent(ABC):
    def __init__(
        self,
        name: str,
        llm_client: LLMClient,
        system_prompt: str,
        tool_registry: ToolRegistry,
        memory_manager: MemoryManager,
        max_iterations: int = 10,
    ):
        self.name = name
        self.llm_client = llm_client
        self.system_prompt = system_prompt
        self.tool_registry = tool_registry
        self.memory = memory_manager
        self.max_iterations = max_iterations

    @abstractmethod
    async def run(self, user_input: str) -> AgentResult:
        ...
