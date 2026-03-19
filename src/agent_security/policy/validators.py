import re

SECRET_PATTERNS = [
    r"BEGIN PRIVATE KEY",
    r"AKIA[0-9A-Z]{16}",
    r"sk-[A-Za-z0-9]{20,}",
    r"[A-Za-z0-9_\-]{32,}",
]

SUSPICIOUS_PATTERNS = [
    r"attacker@",
    r"exfil",
    r"send .* to .*@",
    r"http[s]?://",
]


def validate_tool_args(tool_name: str, args: dict, task_text: str) -> tuple[bool, str]:
    joined = " ".join([str(v) for v in args.values()])

    for p in SECRET_PATTERNS:
        if re.search(p, joined, flags=re.I):
            return False, f"secret_pattern_detected:{p}"

    if tool_name in {"send_email", "share_doc", "post_message", "transfer_money", "delete_email"}:
        if not re.search(r"\b(send|share|post|transfer|delete)\b", task_text, flags=re.I):
            return False, "write_tool_not_explicitly_requested_by_user"

    for p in SUSPICIOUS_PATTERNS:
        if re.search(p, joined, flags=re.I) and tool_name in {"send_email", "share_doc", "post_message"}:
            return False, f"suspicious_exfil_pattern:{p}"

    return True, "ok"
