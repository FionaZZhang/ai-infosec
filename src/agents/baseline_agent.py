"""Baseline agent implementation without any defenses."""

import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from agentdojo.agent_pipeline import AgentPipeline
from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM
from agentdojo.functions_runtime import FunctionsRuntime


@dataclass
class AgentState:
    """Tracks agent state during execution."""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_outputs: List[Dict[str, Any]] = field(default_factory=list)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    final_response: Optional[str] = None


class BaselineAgent:
    """
    Baseline LLM agent with no defensive mechanisms.
    Uses OpenAI function calling with GPT-4o-mini.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.state = AgentState()

    def create_pipeline(self, tools: List[Any]) -> AgentPipeline:
        """Create an AgentDojo pipeline with the given tools."""
        llm = OpenAILLM(
            model=self.model_name,
            temperature=self.temperature,
        )

        pipeline = AgentPipeline(
            llm=llm,
            tools=tools,
        )

        return pipeline

    def reset_state(self):
        """Reset agent state for a new task."""
        self.state = AgentState()

    def log_tool_call(self, tool_name: str, arguments: Dict[str, Any]):
        """Log a tool call."""
        self.state.tool_calls.append({
            "tool": tool_name,
            "arguments": arguments,
        })

    def log_tool_output(self, tool_name: str, output: Any):
        """Log a tool output."""
        self.state.tool_outputs.append({
            "tool": tool_name,
            "output": output,
        })

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the agent's execution."""
        return {
            "total_tool_calls": len(self.state.tool_calls),
            "tools_used": list(set(tc["tool"] for tc in self.state.tool_calls)),
            "total_tool_outputs": len(self.state.tool_outputs),
        }


def create_baseline_pipeline(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> callable:
    """
    Factory function to create a baseline agent pipeline for AgentDojo.

    Returns a function that AgentDojo can use to create pipelines.
    """

    def pipeline_factory() -> AgentPipeline:
        llm = OpenAILLM(
            model=model_name,
            temperature=temperature,
        )
        return AgentPipeline(llm=llm)

    return pipeline_factory
