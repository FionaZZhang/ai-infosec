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
    """Classification of tools by permission level.

    Tool names match AgentDojo benchmark tools:
    - Workspace: email, calendar, cloud drive tools
    - Banking: financial transaction tools
    """

    # Read-only tools (safe, no side effects)
    read_tools: Set[str] = field(default_factory=lambda: {
        # Workspace - Email (AgentDojo names)
        "get_unread_emails", "get_sent_emails", "get_received_emails",
        "get_draft_emails", "search_emails",
        "search_contacts_by_name", "search_contacts_by_email",
        # Workspace - Calendar
        "get_current_day", "search_calendar_events", "get_day_calendar_events",
        # Workspace - Cloud Drive
        "list_files", "search_files", "search_files_by_filename", "get_file_by_id",
        # Banking (AgentDojo names)
        "get_balance", "get_most_recent_transactions", "get_scheduled_transactions",
        "get_iban", "get_user_info", "read_file",
    })

    # Write-capable tools (can modify state)
    write_tools: Set[str] = field(default_factory=lambda: {
        # Workspace - Email
        "send_email", "delete_email",
        # Workspace - Calendar
        "create_calendar_event", "cancel_calendar_event",
        "reschedule_calendar_event", "add_calendar_event_participants",
        # Workspace - Cloud Drive
        "create_file", "delete_file", "append_to_file", "share_file",
        # Banking
        "send_money", "schedule_transaction", "update_scheduled_transaction",
        "update_user_info", "update_password",
    })

    # Sensitive tools (high-risk, require extra confirmation)
    sensitive_tools: Set[str] = field(default_factory=lambda: {
        # High financial risk
        "send_money", "schedule_transaction", "update_scheduled_transaction",
        # External communication / data exfiltration risk
        "send_email", "share_file",
        # Destructive actions
        "delete_email", "delete_file", "cancel_calendar_event",
        # Account security
        "update_password", "update_user_info",
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

    # Model settings (use full model name for AgentDojo compatibility)
    model_name: str = "gpt-4o-mini-2024-07-18"
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
