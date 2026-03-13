"""
Sensitive Tool Gating Defense.

Implements confirmation/step-up authentication for high-risk tool calls.
Write-capable tools require explicit confirmation before execution.
"""

from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

from ..config import (
    ToolClassification,
    DEFAULT_TOOL_CLASSIFICATION,
)


class GatingDecision(Enum):
    """Possible gating decisions."""
    ALLOW = "allow"
    BLOCK = "block"
    REQUIRE_CONFIRMATION = "require_confirmation"
    CONFIRMED = "confirmed"


@dataclass
class GatingResult:
    """Result of a gating decision."""
    tool_name: str
    arguments: Dict[str, Any]
    decision: GatingDecision
    reason: str
    is_attacker_triggered: bool = False


class SensitiveToolGating:
    """
    Implements confirmation gating for sensitive tool calls.

    Key features:
    - Write-capable tools (send_email, transfer_funds, etc.) require confirmation
    - In evaluation mode: auto-confirm for benign tasks, flag attacker-triggered calls
    - Tracks all gating decisions for analysis
    """

    def __init__(
        self,
        tool_classification: Optional[ToolClassification] = None,
        auto_confirm_benign: bool = True,
        confirmation_callback: Optional[Callable[[str, Dict], bool]] = None,
    ):
        """
        Initialize the gating defense.

        Args:
            tool_classification: Tool classification settings
            auto_confirm_benign: Auto-confirm for evaluation (benign tasks)
            confirmation_callback: Custom callback for confirmation prompts
        """
        self.tool_classification = tool_classification or DEFAULT_TOOL_CLASSIFICATION
        self.auto_confirm_benign = auto_confirm_benign
        self.confirmation_callback = confirmation_callback
        self.gating_log: List[GatingResult] = []

        # Track which calls are from the original task vs potentially injected
        self.expected_tool_calls: List[str] = []
        self.is_attack_scenario: bool = False

    def set_attack_scenario(self, is_attack: bool):
        """Set whether this is an attack scenario."""
        self.is_attack_scenario = is_attack

    def set_expected_tools(self, tool_names: List[str]):
        """Set which tools are expected for the legitimate task."""
        self.expected_tool_calls = tool_names

    def check_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> GatingResult:
        """
        Check if a tool call should be allowed, blocked, or confirmed.

        Args:
            tool_name: Name of the tool being called
            arguments: Arguments to the tool call

        Returns:
            GatingResult with the decision
        """
        is_sensitive = self.tool_classification.is_sensitive(tool_name)
        is_write = not self.tool_classification.is_read_only(tool_name)

        # Read-only tools always allowed
        if not is_write:
            result = GatingResult(
                tool_name=tool_name,
                arguments=arguments,
                decision=GatingDecision.ALLOW,
                reason="Read-only tool, no confirmation needed",
            )
            self.gating_log.append(result)
            return result

        # Detect if this might be attacker-triggered
        is_attacker_triggered = self._detect_attacker_triggered(tool_name, arguments)

        # Sensitive tools need special handling
        if is_sensitive:
            if is_attacker_triggered:
                # Block attacker-triggered sensitive calls
                result = GatingResult(
                    tool_name=tool_name,
                    arguments=arguments,
                    decision=GatingDecision.BLOCK,
                    reason="Sensitive tool call appears to be attacker-triggered",
                    is_attacker_triggered=True,
                )
                self.gating_log.append(result)
                return result

            # For evaluation with auto-confirm
            if self.auto_confirm_benign and not self.is_attack_scenario:
                result = GatingResult(
                    tool_name=tool_name,
                    arguments=arguments,
                    decision=GatingDecision.CONFIRMED,
                    reason="Auto-confirmed for benign task evaluation",
                )
                self.gating_log.append(result)
                return result

            # Require confirmation
            confirmed = self._request_confirmation(tool_name, arguments)
            decision = GatingDecision.CONFIRMED if confirmed else GatingDecision.BLOCK

            result = GatingResult(
                tool_name=tool_name,
                arguments=arguments,
                decision=decision,
                reason="Confirmation required for sensitive tool",
            )
            self.gating_log.append(result)
            return result

        # Non-sensitive write tools
        if is_attacker_triggered:
            result = GatingResult(
                tool_name=tool_name,
                arguments=arguments,
                decision=GatingDecision.BLOCK,
                reason="Write tool call appears to be attacker-triggered",
                is_attacker_triggered=True,
            )
            self.gating_log.append(result)
            return result

        result = GatingResult(
            tool_name=tool_name,
            arguments=arguments,
            decision=GatingDecision.ALLOW,
            reason="Write tool allowed for legitimate task",
        )
        self.gating_log.append(result)
        return result

    def _detect_attacker_triggered(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> bool:
        """
        Heuristic detection of attacker-triggered tool calls.

        Looks for signs that a tool call might be from an injection:
        - Tool not in expected list for this task
        - Suspicious patterns in arguments (external URLs, unknown recipients)
        """
        # In attack scenario, check if this tool was expected
        if self.is_attack_scenario:
            if tool_name not in self.expected_tool_calls:
                return True

        # Check for suspicious patterns in arguments
        suspicious_patterns = [
            "attacker",
            "evil",
            "malicious",
            "hack",
            "phishing",
            "exfiltrate",
        ]

        arg_str = str(arguments).lower()
        for pattern in suspicious_patterns:
            if pattern in arg_str:
                return True

        return False

    def _request_confirmation(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> bool:
        """
        Request user confirmation for a tool call.

        Uses custom callback if provided, otherwise auto-confirms
        based on settings.
        """
        if self.confirmation_callback:
            return self.confirmation_callback(tool_name, arguments)

        # Default: auto-confirm if enabled
        return self.auto_confirm_benign

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about gating decisions."""
        if not self.gating_log:
            return {"total_decisions": 0}

        allowed = sum(1 for r in self.gating_log if r.decision == GatingDecision.ALLOW)
        blocked = sum(1 for r in self.gating_log if r.decision == GatingDecision.BLOCK)
        confirmed = sum(1 for r in self.gating_log if r.decision == GatingDecision.CONFIRMED)
        attacker_blocked = sum(1 for r in self.gating_log if r.is_attacker_triggered)

        return {
            "total_decisions": len(self.gating_log),
            "allowed": allowed,
            "blocked": blocked,
            "confirmed": confirmed,
            "attacker_triggered_blocked": attacker_blocked,
        }

    def reset(self):
        """Reset the gating log and state."""
        self.gating_log = []
        self.expected_tool_calls = []
        self.is_attack_scenario = False
