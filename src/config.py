"""Configuration settings for the agent security project."""

from dataclasses import dataclass, field
from typing import Dict, List, Set
from enum import Enum


class ToolCategory(Enum):
    """Categories for tool permissions."""
    READ = "read"
    WRITE = "write"
    SENSITIVE = "sensitive"


@dataclass
class DefenseConfig:
    """Configuration for defense mechanisms."""

    # Tool Permission Scoping
    enable_tool_scoping: bool = True
    default_permission: ToolCategory = ToolCategory.READ

    # Sensitive Tool Gating
    enable_sensitive_gating: bool = True
    require_confirmation: bool = True
    auto_confirm_benign: bool = True  # For evaluation, auto-confirm benign tasks

    # Untrusted Content Quarantine
    enable_quarantine: bool = True
    use_summarizer: bool = True
    max_output_length: int = 2000

    # Dry-run checks (for future extension)
    enable_dry_run: bool = False


@dataclass
class ToolClassification:
    """Classification of tools by permission level."""

    # Read-only tools (safe, no side effects)
    read_tools: Set[str] = field(default_factory=lambda: {
        # Workspace/Email
        "search_emails", "get_email_by_id", "list_emails",
        "search_contacts", "get_contact_by_id", "list_contacts",
        "get_calendar_events", "search_calendar",
        "get_file", "list_files", "search_files",
        # Banking
        "get_balance", "get_transactions", "get_scheduled_transactions",
        "get_account_info", "get_user_info", "get_iban",
        # General
        "get_current_time", "get_user_preferences",
    })

    # Write-capable tools (can modify state)
    write_tools: Set[str] = field(default_factory=lambda: {
        # Workspace/Email
        "send_email", "reply_to_email", "forward_email",
        "create_contact", "update_contact", "delete_contact",
        "create_calendar_event", "update_calendar_event", "delete_calendar_event",
        "create_file", "update_file", "delete_file",
        "share_file", "invite_to_event",
        # Banking
        "transfer_money", "schedule_transaction", "cancel_transaction",
        "update_scheduled_transaction",
    })

    # Sensitive tools (high-risk, require extra confirmation)
    sensitive_tools: Set[str] = field(default_factory=lambda: {
        # High financial risk
        "transfer_money", "schedule_transaction",
        # External communication
        "send_email", "forward_email", "share_file",
        # Data exposure
        "invite_to_event",
    })

    def get_category(self, tool_name: str) -> ToolCategory:
        """Get the category of a tool."""
        if tool_name in self.sensitive_tools:
            return ToolCategory.SENSITIVE
        elif tool_name in self.write_tools:
            return ToolCategory.WRITE
        elif tool_name in self.read_tools:
            return ToolCategory.READ
        else:
            # Unknown tools default to WRITE for safety
            return ToolCategory.WRITE

    def is_read_only(self, tool_name: str) -> bool:
        """Check if tool is read-only."""
        return tool_name in self.read_tools

    def is_sensitive(self, tool_name: str) -> bool:
        """Check if tool is sensitive."""
        return tool_name in self.sensitive_tools


@dataclass
class TaskClassification:
    """Keywords for classifying tasks as read-only vs read-write."""

    read_only_keywords: List[str] = field(default_factory=lambda: [
        "summarize", "list", "find", "search", "get", "check",
        "show", "display", "read", "view", "tell me", "what is",
        "how many", "when is", "who is", "look up", "retrieve",
    ])

    write_keywords: List[str] = field(default_factory=lambda: [
        "send", "create", "update", "delete", "schedule", "transfer",
        "reply", "forward", "share", "invite", "book", "cancel",
        "modify", "change", "set", "add", "remove", "make",
    ])

    def classify_task(self, task_description: str) -> ToolCategory:
        """Classify a task based on its description."""
        task_lower = task_description.lower()

        # Check for write keywords first (more restrictive)
        for keyword in self.write_keywords:
            if keyword in task_lower:
                return ToolCategory.WRITE

        # Then check for read-only keywords
        for keyword in self.read_only_keywords:
            if keyword in task_lower:
                return ToolCategory.READ

        # Default to WRITE for safety
        return ToolCategory.WRITE


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs."""

    # Model settings
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 4096

    # Evaluation settings
    task_suites: List[str] = field(default_factory=lambda: [
        "workspace",
        "banking",
    ])

    # Attack settings
    attack_type: str = "important_instructions"  # AgentDojo default

    # Output settings
    results_dir: str = "results"
    save_traces: bool = True
    verbose: bool = True


# Default configurations
DEFAULT_DEFENSE_CONFIG = DefenseConfig()
DEFAULT_TOOL_CLASSIFICATION = ToolClassification()
DEFAULT_TASK_CLASSIFICATION = TaskClassification()
DEFAULT_EVAL_CONFIG = EvaluationConfig()
