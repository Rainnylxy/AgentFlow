"""Tests for the Handoff protocol data structures."""
from agentflow.runtime.handoff import HandoffRequest, RouteResult, parse_handoff_block


class TestHandoffRequest:
    def test_create_full(self):
        hr = HandoffRequest(
            reason="Not my domain",
            suggested_agent="payment_agent",
            suggested_since="payments are payment_agent's specialty",
            partial_result="I found the order but can't process the payment.",
        )
        assert hr.reason == "Not my domain"
        assert hr.suggested_agent == "payment_agent"
        assert hr.partial_result != ""

    def test_create_minimal(self):
        hr = HandoffRequest(
            reason="Out of scope",
            suggested_agent="",
            suggested_since="",
            partial_result="",
        )
        assert hr.reason == "Out of scope"

    def test_suggested_agent_is_optional(self):
        hr = HandoffRequest(reason="Can't do this")
        assert hr.suggested_agent == ""


class TestRouteResult:
    def test_complete_no_handoff(self):
        rr = RouteResult(
            agent_id="refund_agent",
            output="Your refund has been processed.",
            tool_calls=[
                {"tool": "process_refund", "input": {"id": "123"}, "output": "success"},
            ],
            handoff=None,
        )
        assert rr.agent_id == "refund_agent"
        assert rr.handoff is None

    def test_with_handoff(self):
        ho = HandoffRequest(
            reason="Cross-border not supported",
            suggested_agent="cross_border_agent",
            suggested_since="They handle international",
            partial_result="Order #456 is valid, amount is $100 USD.",
        )
        rr = RouteResult(
            agent_id="refund_agent",
            output="",
            tool_calls=[],
            handoff=ho,
        )
        assert rr.handoff is ho
        assert rr.handoff.reason == "Cross-border not supported"


class TestParseHandoffBlock:
    def test_parse_valid_block(self):
        text = """I tried to help but this is out of my scope.

---HANDOFF---
reason: Cross-border payments are not covered by my domain
suggest: An agent that handles international payment processing
context: The user is trying to send $500 to a UK bank account. Their account is verified.
---END---

Let me know if you need more help."""
        ho = parse_handoff_block(text)
        assert ho is not None
        assert "Cross-border payments" in ho.reason
        assert "international payment" in ho.suggested_agent
        assert "$500" in ho.partial_result

    def test_parse_no_handoff_block(self):
        text = "Your refund has been processed successfully. Is there anything else?"
        ho = parse_handoff_block(text)
        assert ho is None

    def test_parse_malformed_block_returns_none(self):
        text = """---HANDOFF---
incomplete block no end marker"""
        ho = parse_handoff_block(text)
        assert ho is None

    def test_parse_block_with_extra_whitespace(self):
        text = """
---HANDOFF---
reason:   Too complex
suggest:   senior_agent

context:   Need help with advanced math
---END---
"""
        ho = parse_handoff_block(text)
        assert ho is not None
        assert ho.reason == "Too complex"
        assert ho.suggested_agent == "senior_agent"
        assert "advanced math" in ho.partial_result
