import pytest
from agentflow.runtime.memory.semantic import SemanticMemory


class TestSemanticMemory:
    def test_store_and_search_basic(self):
        mem = SemanticMemory()
        mem.store("key1", "AgentFlow is a Go+Python framework")
        mem.store("key2", "LangChain is a Python LLM framework")

        results = mem.search("Go")
        assert len(results) == 1
        assert "AgentFlow" in str(results[0])

    def test_get_missing(self):
        mem = SemanticMemory()
        assert mem.get("nonexistent") is None

    def test_top_k_limit(self):
        mem = SemanticMemory()
        for i in range(10):
            mem.store(f"key{i}", f"Document number {i} containing keyword")
        results = mem.search("keyword", top_k=3)
        assert len(results) <= 3

    def test_search_no_match(self):
        mem = SemanticMemory()
        mem.store("k1", "Weather in Beijing")
        results = mem.search("quantum")
        assert len(results) == 0
