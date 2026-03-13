"""
Policy & Isolation Layer.

Combines all defense mechanisms into a unified layer that can be
integrated with the agent pipeline.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .tool_scoping import ToolScopingDefense, ScopingDecision
from .sensitive_gating import SensitiveToolGating, GatingResult, GatingDecision
from .quarantine import QuarantineBuffer, ContentSummarizer, QuarantinedContent
from ..config import (
    DefenseConfig,
    ToolCategory,
    DEFAULT_DEFENSE_CONFIG,
)


@dataclass
class PolicyDecision:
    """Complete policy decision for a tool interaction."""
    tool_name: str
    action: str  # "allow", "block", "quarantine"
    scoping_decision: Optional[ScopingDecision] = None
    gating_result: Optional[GatingResult] = None
    quarantined_content: Optional[QuarantinedContent] = None
    sanitized_output: Optional[str] = None
    reasons: List[str] = field(default_factory=list)


class PolicyAndIsolationLayer:
    """
    Complete Policy & Isolation Layer integrating all defense mechanisms.

    Flow:
    1. Task arrives -> Tool Scoping determines allowed tools
    2. Tool call proposed -> Gating checks if call should proceed
    3. Tool executed -> Output quarantined and summarized
    4. Sanitized output returned to agent context
    """

    def __init__(
        self,
        config: Optional[DefenseConfig] = None,
        llm_client: Optional[Any] = None,
    ):
        self.config = config or DEFAULT_DEFENSE_CONFIG

        # Initialize defense components
        self.tool_scoping = ToolScopingDefense()
        self.gating = SensitiveToolGating(
            auto_confirm_benign=self.config.auto_confirm_benign,
        )
        self.quarantine = QuarantineBuffer()
        self.summarizer = ContentSummarizer(
            llm_client=llm_client,
            max_output_length=self.config.max_output_length,
            use_llm_summarizer=self.config.use_summarizer,
        )

        # Current task state
        self.current_task: Optional[str] = None
        self.current_task_category: Optional[ToolCategory] = None
        self.allowed_tools: set = set()

        # Decision log
        self.decision_log: List[PolicyDecision] = []

        # Metrics
        self.metrics = {
            "total_tool_calls": 0,
            "calls_allowed": 0,
            "calls_blocked": 0,
            "injections_detected": 0,
            "attacker_calls_blocked": 0,
        }

    def begin_task(
        self,
        task_description: str,
        available_tools: List[str],
        is_attack_scenario: bool = False,
        expected_tools: Optional[List[str]] = None,
    ) -> ScopingDecision:
        """
        Begin processing a new task.

        This is called at the start of a task, before the agent
        sees any tool outputs.

        Args:
            task_description: The user's task request
            available_tools: All available tool names
            is_attack_scenario: Whether this is an attack evaluation
            expected_tools: Tools expected for legitimate task completion

        Returns:
            ScopingDecision with allowed tools
        """
        self.current_task = task_description

        # Tool scoping (if enabled)
        if self.config.enable_tool_scoping:
            scoping = self.tool_scoping.get_allowed_tools(
                task_description,
                available_tools,
            )
            self.current_task_category = scoping.task_category
            self.allowed_tools = scoping.allowed_tools
        else:
            self.current_task_category = ToolCategory.WRITE
            self.allowed_tools = set(available_tools)
            scoping = ScopingDecision(
                task_category=ToolCategory.WRITE,
                allowed_tools=self.allowed_tools,
                blocked_tools=set(),
                reason="Tool scoping disabled",
            )

        # Configure gating
        self.gating.set_attack_scenario(is_attack_scenario)
        if expected_tools:
            self.gating.set_expected_tools(expected_tools)

        return scoping

    def check_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> PolicyDecision:
        """
        Check if a tool call should be allowed.

        Args:
            tool_name: Name of the tool being called
            arguments: Arguments to the tool

        Returns:
            PolicyDecision with the verdict
        """
        self.metrics["total_tool_calls"] += 1
        decision = PolicyDecision(
            tool_name=tool_name,
            action="allow",
        )

        # Check tool scoping
        if self.config.enable_tool_scoping:
            if tool_name not in self.allowed_tools:
                decision.action = "block"
                decision.reasons.append(
                    f"Tool '{tool_name}' not in allowed list for this task"
                )
                self.metrics["calls_blocked"] += 1
                self.decision_log.append(decision)
                return decision

        # Check sensitive gating
        if self.config.enable_sensitive_gating:
            gating_result = self.gating.check_tool_call(tool_name, arguments)
            decision.gating_result = gating_result

            if gating_result.decision == GatingDecision.BLOCK:
                decision.action = "block"
                decision.reasons.append(gating_result.reason)
                self.metrics["calls_blocked"] += 1

                if gating_result.is_attacker_triggered:
                    self.metrics["attacker_calls_blocked"] += 1

                self.decision_log.append(decision)
                return decision

        self.metrics["calls_allowed"] += 1
        decision.reasons.append("Tool call allowed")
        self.decision_log.append(decision)
        return decision

    def process_tool_output(
        self,
        tool_name: str,
        raw_output: Any,
    ) -> Tuple[str, QuarantinedContent]:
        """
        Process tool output through quarantine and summarization.

        Args:
            tool_name: Name of the tool that produced the output
            raw_output: Raw output from the tool

        Returns:
            (sanitized_output, quarantined_content)
        """
        # Add to quarantine buffer
        if self.config.enable_quarantine:
            content = self.quarantine.add(tool_name, raw_output)

            if content.injection_detected:
                self.metrics["injections_detected"] += 1

            # Summarize to extract data only
            sanitized = self.summarizer.summarize(content, self.current_task)
        else:
            content = QuarantinedContent(
                tool_name=tool_name,
                raw_output=raw_output,
            )
            sanitized = str(raw_output)

        return sanitized, content

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "policy_layer": self.metrics,
            "tool_scoping": self.tool_scoping.get_stats(),
            "gating": self.gating.get_stats(),
            "quarantine": self.quarantine.get_stats(),
            "summarizer": self.summarizer.get_stats(),
        }

    def reset(self):
        """Reset the policy layer for a new task."""
        self.current_task = None
        self.current_task_category = None
        self.allowed_tools = set()
        self.decision_log = []

        # Reset components
        self.tool_scoping.reset()
        self.gating.reset()
        self.quarantine.clear()
        self.summarizer.reset()

        # Reset metrics
        self.metrics = {
            "total_tool_calls": 0,
            "calls_allowed": 0,
            "calls_blocked": 0,
            "injections_detected": 0,
            "attacker_calls_blocked": 0,
        }

    def get_decision_summary(self) -> Dict[str, Any]:
        """Get a summary of all decisions made."""
        blocked_tools = [
            d.tool_name for d in self.decision_log
            if d.action == "block"
        ]
        allowed_tools = [
            d.tool_name for d in self.decision_log
            if d.action == "allow"
        ]

        return {
            "total_decisions": len(self.decision_log),
            "allowed_tools": allowed_tools,
            "blocked_tools": blocked_tools,
            "task_category": self.current_task_category.value if self.current_task_category else None,
        }
