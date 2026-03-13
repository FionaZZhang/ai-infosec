"""
Tool Permission Scoping Defense.

Implements least-privilege tool access by classifying tasks and restricting
available tools to only those necessary for the task.

This is a NON-LLM rule engine to prevent attacker influence on the gate itself.
"""

from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field

from ..config import (
    ToolCategory,
    ToolClassification,
    TaskClassification,
    DEFAULT_TOOL_CLASSIFICATION,
    DEFAULT_TASK_CLASSIFICATION,
)


@dataclass
class ScopingDecision:
    """Result of tool scoping decision."""
    task_category: ToolCategory
    allowed_tools: Set[str]
    blocked_tools: Set[str]
    reason: str


class ToolScopingDefense:
    """
    Implements tool permission scoping based on task classification.

    Key features:
    - Deterministic rule-based classification (not LLM-based)
    - Pre-planning step that runs before agent sees any tool output
    - Enforces least-privilege access
    """

    def __init__(
        self,
        tool_classification: Optional[ToolClassification] = None,
        task_classification: Optional[TaskClassification] = None,
    ):
        self.tool_classification = tool_classification or DEFAULT_TOOL_CLASSIFICATION
        self.task_classification = task_classification or DEFAULT_TASK_CLASSIFICATION
        self.scoping_log: List[ScopingDecision] = []

    def classify_task(self, task_description: str) -> ToolCategory:
        """
        Classify task as read-only or read-write based on keywords.

        Uses deterministic rule matching, NOT an LLM, to prevent
        adversarial task descriptions from tricking the classifier.
        """
        return self.task_classification.classify_task(task_description)

    def get_allowed_tools(
        self,
        task_description: str,
        available_tools: List[str],
    ) -> ScopingDecision:
        """
        Determine which tools should be allowed for a given task.

        Args:
            task_description: The user's task request
            available_tools: List of all available tool names

        Returns:
            ScopingDecision with allowed and blocked tools
        """
        task_category = self.classify_task(task_description)

        allowed_tools = set()
        blocked_tools = set()

        for tool in available_tools:
            tool_category = self.tool_classification.get_category(tool)

            if task_category == ToolCategory.READ:
                # Read-only tasks: only allow read tools
                if tool_category == ToolCategory.READ:
                    allowed_tools.add(tool)
                else:
                    blocked_tools.add(tool)
            else:
                # Write tasks: allow all tools
                allowed_tools.add(tool)

        reason = (
            f"Task classified as {task_category.value}. "
            f"Allowed {len(allowed_tools)} tools, blocked {len(blocked_tools)}."
        )

        decision = ScopingDecision(
            task_category=task_category,
            allowed_tools=allowed_tools,
            blocked_tools=blocked_tools,
            reason=reason,
        )

        self.scoping_log.append(decision)
        return decision

    def filter_tools(
        self,
        tools: List[Any],
        allowed_tool_names: Set[str],
    ) -> List[Any]:
        """
        Filter a list of tool objects to only include allowed tools.

        Args:
            tools: List of tool objects (with .name attribute or similar)
            allowed_tool_names: Set of allowed tool names

        Returns:
            Filtered list of tools
        """
        filtered = []
        for tool in tools:
            # Handle different tool object formats
            tool_name = getattr(tool, "name", None) or getattr(tool, "__name__", str(tool))
            if tool_name in allowed_tool_names:
                filtered.append(tool)
        return filtered

    def should_block_tool_call(
        self,
        tool_name: str,
        task_category: ToolCategory,
    ) -> tuple[bool, str]:
        """
        Check if a specific tool call should be blocked.

        This is an additional runtime check for tool calls that
        might slip through.

        Returns:
            (should_block, reason)
        """
        tool_category = self.tool_classification.get_category(tool_name)

        if task_category == ToolCategory.READ:
            if tool_category in (ToolCategory.WRITE, ToolCategory.SENSITIVE):
                return True, f"Tool '{tool_name}' is not allowed for read-only tasks"

        return False, ""

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about scoping decisions."""
        if not self.scoping_log:
            return {"total_decisions": 0}

        read_only_tasks = sum(
            1 for d in self.scoping_log
            if d.task_category == ToolCategory.READ
        )
        write_tasks = sum(
            1 for d in self.scoping_log
            if d.task_category in (ToolCategory.WRITE, ToolCategory.SENSITIVE)
        )
        total_blocked = sum(len(d.blocked_tools) for d in self.scoping_log)

        return {
            "total_decisions": len(self.scoping_log),
            "read_only_tasks": read_only_tasks,
            "write_tasks": write_tasks,
            "total_tools_blocked": total_blocked,
        }

    def reset(self):
        """Reset the scoping log."""
        self.scoping_log = []
