from dataclasses import dataclass
from .tool_risk import risk_of, READ_TOOLS
from .validators import validate_tool_args
from .semantic_guard import evaluate_tool_call


@dataclass
class PolicyDecision:
    allow: bool
    reason: str
    need_confirmation: bool = False


class PolicyEngine:
    def __init__(
        self,
        deny_high_risk: bool = True,
        semantic_backend: str = "heuristic",
        semantic_model_name: str = "gemini-2.5-flash",
        semantic_threshold: float = 0.6,
    ):
        self.deny_high_risk = deny_high_risk
        self.semantic_backend = semantic_backend
        self.semantic_model_name = semantic_model_name
        self.semantic_threshold = semantic_threshold

    def allowed_tools_for_task(self, task_text: str) -> set[str]:
        t = task_text.lower()
        tools = set(READ_TOOLS)
        # Explicit write intent from user allows proposal of send_email,
        # later gated by high-risk checks + semantic guard.
        if "send" in t and "email" in t:
            tools.add("send_email")
        return tools

    def check_tool_call(self, task_text: str, tool_name: str, args: dict) -> PolicyDecision:
        allowed = self.allowed_tools_for_task(task_text)
        if tool_name not in allowed:
            return PolicyDecision(False, f"tool_not_in_allowlist:{tool_name}")

        risk = risk_of(tool_name)

        ok, reason = validate_tool_args(tool_name, args, task_text)
        if not ok:
            return PolicyDecision(False, reason)

        if risk == "HIGH":
            # High-risk actions must pass semantic guard (heuristic or gemini).
            sg = evaluate_tool_call(
                task_text,
                tool_name,
                args,
                backend=self.semantic_backend,
                model_name=self.semantic_model_name,
            )
            if sg.get("allow", False) and float(sg.get("confidence", 0.0)) >= self.semantic_threshold:
                return PolicyDecision(True, "high_risk_semantic_allow")
            return PolicyDecision(False, str(sg.get("reason", f"high_risk_tool_denied:{tool_name}")))

        # For low-risk tools, do not apply semantic guard by default to avoid
        # false positives that hurt utility.
        return PolicyDecision(True, "allowed")
