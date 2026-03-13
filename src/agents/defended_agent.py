"""Defended agent implementation with Policy & Isolation Layer."""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from agentdojo.agent_pipeline import AgentPipeline
from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM

from ..defenses.policy_layer import PolicyAndIsolationLayer, PolicyDecision
from ..config import DefenseConfig, DEFAULT_DEFENSE_CONFIG


@dataclass
class DefendedAgentState:
    """Tracks defended agent state during execution."""
    tool_calls_attempted: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls_allowed: List[Dict[str, Any]] = field(default_factory=list)
    tool_calls_blocked: List[Dict[str, Any]] = field(default_factory=list)
    policy_decisions: List[PolicyDecision] = field(default_factory=list)
    sanitized_outputs: List[str] = field(default_factory=list)
    final_response: Optional[str] = None


class DefendedAgent:
    """
    LLM agent with Policy & Isolation Layer defenses.

    Implements:
    1. Tool permission scoping (least privilege)
    2. Sensitive tool gating (confirmation for write tools)
    3. Untrusted content quarantine (summarize before returning)
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        defense_config: Optional[DefenseConfig] = None,
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.defense_config = defense_config or DEFAULT_DEFENSE_CONFIG
        self.api_key = api_key

        # Initialize defense layer
        self.policy_layer = PolicyAndIsolationLayer(
            config=self.defense_config,
        )

        self.state = DefendedAgentState()

    def begin_task(
        self,
        task_description: str,
        available_tools: List[str],
        is_attack_scenario: bool = False,
        expected_tools: Optional[List[str]] = None,
    ):
        """
        Initialize for a new task.

        Called before the agent starts processing to set up
        tool scoping and other defenses.
        """
        self.reset_state()

        scoping = self.policy_layer.begin_task(
            task_description=task_description,
            available_tools=available_tools,
            is_attack_scenario=is_attack_scenario,
            expected_tools=expected_tools,
        )

        return scoping

    def check_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> PolicyDecision:
        """
        Check if a tool call should be allowed.

        Returns the policy decision.
        """
        self.state.tool_calls_attempted.append({
            "tool": tool_name,
            "arguments": arguments,
        })

        decision = self.policy_layer.check_tool_call(tool_name, arguments)
        self.state.policy_decisions.append(decision)

        if decision.action == "allow":
            self.state.tool_calls_allowed.append({
                "tool": tool_name,
                "arguments": arguments,
            })
        else:
            self.state.tool_calls_blocked.append({
                "tool": tool_name,
                "arguments": arguments,
                "reason": decision.reasons,
            })

        return decision

    def process_output(
        self,
        tool_name: str,
        raw_output: Any,
    ) -> str:
        """
        Process tool output through quarantine and summarization.

        Returns sanitized output safe to pass to agent context.
        """
        sanitized, _ = self.policy_layer.process_tool_output(
            tool_name=tool_name,
            raw_output=raw_output,
        )

        self.state.sanitized_outputs.append(sanitized)
        return sanitized

    def reset_state(self):
        """Reset agent state for a new task."""
        self.state = DefendedAgentState()
        self.policy_layer.reset()

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "agent_state": {
                "tool_calls_attempted": len(self.state.tool_calls_attempted),
                "tool_calls_allowed": len(self.state.tool_calls_allowed),
                "tool_calls_blocked": len(self.state.tool_calls_blocked),
            },
            "policy_layer": self.policy_layer.get_stats(),
        }

    def get_allowed_tools(self) -> set:
        """Get the set of currently allowed tools."""
        return self.policy_layer.allowed_tools


def create_defense_config(
    enable_tool_scoping: bool = True,
    enable_sensitive_gating: bool = True,
    enable_quarantine: bool = True,
) -> DefenseConfig:
    """
    Factory function to create defense configurations.

    Useful for ablation studies.
    """
    return DefenseConfig(
        enable_tool_scoping=enable_tool_scoping,
        enable_sensitive_gating=enable_sensitive_gating,
        enable_quarantine=enable_quarantine,
    )
