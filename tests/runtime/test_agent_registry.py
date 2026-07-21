"""Tests for AgentCapability and AgentRegistry."""
from agentflow.runtime.agent_registry import AgentCapability, AgentRegistry


class TestAgentCapability:
    def test_create_capability(self):
        cap = AgentCapability(
            agent_id="billing_agent",
            description="Handles refund, billing, payment disputes",
            tools=["get_invoice", "process_refund"],
            examples=["customer wants a refund", "billing inquiry"],
            priority=1,
        )
        assert cap.agent_id == "billing_agent"
        assert cap.description == "Handles refund, billing, payment disputes"
        assert cap.tools == ["get_invoice", "process_refund"]
        assert cap.examples == ["customer wants a refund", "billing inquiry"]
        assert cap.priority == 1

    def test_default_priority(self):
        cap = AgentCapability(
            agent_id="default_agent",
            description="General purpose agent",
        )
        assert cap.priority == 0
        assert cap.tools == []
        assert cap.examples == []


class TestAgentRegistry:
    def test_register_and_get(self):
        reg = AgentRegistry()
        cap = AgentCapability(agent_id="billing", description="Handles billing")
        reg.register(cap)
        assert reg.get("billing") is cap

    def test_register_duplicate_overwrites(self):
        reg = AgentRegistry()
        cap1 = AgentCapability(agent_id="billing", description="Handles billing")
        cap2 = AgentCapability(agent_id="billing", description="Handles payments")
        reg.register(cap1)
        reg.register(cap2)
        assert reg.get("billing") is cap2

    def test_unregister(self):
        reg = AgentRegistry()
        cap = AgentCapability(agent_id="billing", description="Handles billing")
        reg.register(cap)
        reg.unregister("billing")
        assert reg.get("billing") is None

    def test_unregister_nonexistent_no_error(self):
        reg = AgentRegistry()
        # Must not raise
        reg.unregister("ghost")

    def test_match_returns_top_candidates(self):
        reg = AgentRegistry()
        reg.register(
            AgentCapability(agent_id="billing", description="billing refund payment")
        )
        reg.register(
            AgentCapability(agent_id="support", description="technical support troubleshooting")
        )
        reg.register(
            AgentCapability(agent_id="general", description="general information help")
        )

        results = reg.match("billing support", top_k=2)
        assert len(results) == 2
        # billing and support both matched something
        ids = {r[0].agent_id for r in results}
        assert "billing" in ids
        assert "support" in ids
        for _, score in results:
            assert score > 0

    def test_match_no_match_returns_empty(self):
        reg = AgentRegistry()
        reg.register(
            AgentCapability(agent_id="billing", description="billing refund payment")
        )
        results = reg.match("xyzzy unknown term")
        assert results == []

    def test_match_single_candidate(self):
        reg = AgentRegistry()
        reg.register(
            AgentCapability(agent_id="billing", description="billing refund payment")
        )
        results = reg.match("billing")
        assert len(results) == 1
        assert results[0][0].agent_id == "billing"
        assert results[0][1] > 0

    def test_match_priority_boost(self):
        reg = AgentRegistry()
        low = AgentCapability(
            agent_id="low", description="billing refund", priority=0
        )
        high = AgentCapability(
            agent_id="high", description="billing refund", priority=50
        )
        reg.register(low)
        reg.register(high)

        results = reg.match("billing refund")
        assert len(results) == 2
        # high priority should rank first
        assert results[0][0].agent_id == "high"
        # score of high should be > low (same base jaccard, higher priority boost)
        assert results[0][1] > results[1][1]

    def test_match_request_more_than_registered(self):
        reg = AgentRegistry()
        reg.register(
            AgentCapability(agent_id="billing", description="billing refund payment")
        )
        results = reg.match("billing", top_k=10)
        assert len(results) == 1
