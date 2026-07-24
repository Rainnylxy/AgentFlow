"""RoutingStrategy — dynamic expert routing with handoff loop.

Selects the best expert from registry matches using the LLM,
executes the expert directly, and handles handoff signals.

State machine:
    ANALYZE -> ROUTE -> EXECUTE -> CHECK_HANDOFF
       ^                            |
       |        (handoff)           |
       +--- back to ANALYZE --------+
"""

from __future__ import annotations

import json
import time
from typing import Optional

from agentflow.runtime.agent import AgentResult
from agentflow.runtime.agent_registry import AgentCapability, AgentRegistry
from agentflow.runtime.handoff import HandoffRequest, parse_handoff_block
from agentflow.runtime.hooks import StreamEvent
from agentflow.runtime.thinking.base import ThinkingStrategy, ThinkContext, ThinkResult


class RoutingStrategy(ThinkingStrategy):
    """Dynamic expert routing strategy.

    Maintains a state machine that analyzes the task, routes to the best
    expert, executes the expert, and checks for handoff signals. If a
    handoff is detected, the loop repeats with accumulated context.

    Args:
        registry: AgentRegistry with capability descriptions for matching.
        experts: Dict mapping agent_id -> expert instance (must have
                 async ``run(user_input, stream, agent_trace)`` method
                 returning ``AgentResult``).
        toolkit: Optional ToolKit for shared tools.
        max_handoffs: Max handoff cycles before forced termination (default 3).
    """

    def __init__(
        self,
        registry: AgentRegistry,
        experts: dict,
        toolkit=None,
        max_handoffs: int = 3,
    ):
        super().__init__(toolkit=toolkit)
        self.registry = registry
        self.experts = experts
        self.max_handoffs = max_handoffs

    async def run(self, context: ThinkContext) -> ThinkResult:
        """Run the ANALYZE -> ROUTE -> EXECUTE -> CHECK_HANDOFF loop."""
        tried_agents: set[str] = set()
        all_steps: list[dict] = []
        all_tool_calls: list = []
        handoff_chain: list[HandoffRequest] = []
        last_output = ""

        for cycle in range(self.max_handoffs + 1):
            # ---- ANALYZE ----
            candidates = self.registry.match(context.user_input, top_k=3)
            if not candidates:
                return ThinkResult(
                    output="I don't have an expert capable of handling this task.",
                    tool_calls=all_tool_calls,
                    steps=all_steps,
                    mode_used="routing",
                )

            # ---- ROUTE ----
            # Filter out already-tried agents
            available = [
                (cap, score)
                for cap, score in candidates
                if cap.agent_id not in tried_agents
            ]
            if not available:
                note = (
                    "All available agents have been tried without completing "
                    f"the task. Tried agents: {', '.join(sorted(tried_agents))}"
                )
                return ThinkResult(
                    output=note,
                    tool_calls=all_tool_calls,
                    steps=all_steps,
                    mode_used="routing",
                )

            selected_id = await self._route_to_expert(
                context, available, handoff_chain,
            )
            if selected_id is None:
                # LLM routing failed, fallback to top-scoring candidate
                selected_id = available[0][0].agent_id

            await context.emit(StreamEvent(
                type="route_decision",
                content=f"Routing to expert: {selected_id}",
                data={"agent_id": selected_id},
            ))

            tried_agents.add(selected_id)

            # ---- EXECUTE ----
            expert = self.experts.get(selected_id)
            if expert is None:
                await context.emit(StreamEvent(
                    type="expert_error",
                    content=f"Expert '{selected_id}' not found in experts dict.",
                    data={"agent_id": selected_id},
                ))
                continue

            await context.emit(StreamEvent(
                type="expert_start",
                content=f"Starting expert: {selected_id}",
                data={"agent_id": selected_id},
            ))

            t_start = time.monotonic()
            try:
                task_context = self._build_task_context(
                    context.user_input, handoff_chain,
                )
                agent_result: AgentResult = await expert.run(
                    user_input=task_context,
                    stream=context.stream,
                    agent_trace=context.agent_trace,
                )
                t_dur = int((time.monotonic() - t_start) * 1000)

                await context.emit(StreamEvent(
                    type="expert_done",
                    content=f"Expert {selected_id} completed.",
                    data={"agent_id": selected_id, "duration_ms": t_dur},
                ))

                step = {
                    "agent_id": selected_id,
                    "output": agent_result.output,
                    "tool_calls": agent_result.tool_calls,
                    "duration_ms": t_dur,
                }
                all_steps.append(step)
                all_tool_calls.extend(agent_result.tool_calls)
                last_output = agent_result.output

            except Exception as e:
                t_dur = int((time.monotonic() - t_start) * 1000)
                await context.emit(StreamEvent(
                    type="expert_error",
                    content=f"Expert {selected_id} failed: {e}",
                    data={"agent_id": selected_id, "error": str(e)},
                ))
                all_steps.append({
                    "agent_id": selected_id,
                    "output": f"ERROR: {e}",
                    "tool_calls": [],
                    "duration_ms": t_dur,
                })
                continue

            # ---- CHECK_HANDOFF ----
            handoff = parse_handoff_block(agent_result.output)
            if handoff is None:
                # No handoff -- task is complete
                return ThinkResult(
                    output=agent_result.output,
                    tool_calls=all_tool_calls,
                    steps=all_steps,
                    mode_used="routing",
                )

            # Handoff detected
            handoff_chain.append(handoff)
            await context.emit(StreamEvent(
                type="handoff",
                content=f"Handoff from {selected_id}: {handoff.reason}",
                data={
                    "from_agent": selected_id,
                    "handoff": {
                        "reason": handoff.reason,
                        "suggested_agent": handoff.suggested_agent,
                        "partial_result": handoff.partial_result,
                    },
                },
            ))

        # Handoff loop exhausted
        note = (
            f"Handoff limit ({self.max_handoffs}) reached. "
            f"Tried agents: {', '.join(sorted(tried_agents))}. "
            f"Last output: {last_output[:200]}"
        )
        return ThinkResult(
            output=note,
            tool_calls=all_tool_calls,
            steps=all_steps,
            mode_used="routing",
        )

    async def _route_to_expert(
        self,
        context: ThinkContext,
        candidates: list[tuple[AgentCapability, float]],
        handoff_chain: list[HandoffRequest],
    ) -> Optional[str]:
        """Ask the LLM to select the best expert from candidates.

        Returns the selected agent_id, or None if parsing fails (caller
        should fall back to the top-scoring candidate).
        """
        candidates_text = "\n".join(
            f"- {cap.agent_id}: {cap.description} (score: {score:.3f})"
            for cap, score in candidates
        )

        handoff_context = ""
        if handoff_chain:
            parts = ["\nPrevious handoff context:"]
            for h in handoff_chain:
                parts.append(f"- {h.reason}")
            handoff_context = "\n".join(parts)

        system_prompt = (
            "You are a router that selects the best expert agent for a task.\n"
            "Analyze the task and the available candidates, then select the "
            "most capable expert. Respond in JSON format only:\n"
            '{"agent_id": "<selected agent_id>", "reason": "<why this agent>"}'
        )

        user_prompt = (
            f"Task: {context.user_input}\n\n"
            f"Available candidates:\n{candidates_text}"
            f"{handoff_context}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = await context.llm_client.chat(
                messages, max_tokens=context.max_output_tokens,
            )
            content = response.content or ""
            parsed = self._parse_json_response(content)
            if parsed is not None and "agent_id" in parsed:
                agent_id = parsed["agent_id"]
                # Verify the agent_id is actually in candidates
                valid_ids = {cap.agent_id for cap, _ in candidates}
                if agent_id in valid_ids:
                    return agent_id
                # Case-insensitive fallback
                for cap, _ in candidates:
                    if cap.agent_id.lower() == agent_id.lower():
                        return cap.agent_id
        except Exception:
            pass

        return None

    @staticmethod
    def _parse_json_response(content: str) -> Optional[dict]:
        """Parse JSON from LLM response, handling markdown code fences."""
        content = content.strip()
        # Strip markdown code fences if present
        if content.startswith("```"):
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1:
                content = content[start:end + 1]
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _build_task_context(
        user_input: str,
        handoff_chain: list[HandoffRequest],
    ) -> str:
        """Build task context string with accumulated handoff information."""
        if not handoff_chain:
            return user_input

        parts = [f"Original task: {user_input}"]
        for h in handoff_chain:
            ctx_parts = []
            if h.partial_result:
                ctx_parts.append(f"Previous work: {h.partial_result}")
            if h.suggested_agent:
                ctx_parts.append(f"Suggested next agent: {h.suggested_agent}")
            ctx_parts.append(f"Handoff reason: {h.reason}")
            parts.extend(ctx_parts)

        return "\n\n".join(parts)
