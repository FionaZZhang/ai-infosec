import re

INSTRUCTIONY = [
    r"ignore previous",
    r"send email",
    r"wire transfer",
    r"do not tell",
    r"click",
    r"execute",
]


def detect_injection(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in INSTRUCTIONY)


def strip_instruction_lines(text: str) -> tuple[str, list[str]]:
    kept, dropped = [], []
    for line in text.splitlines():
        if detect_injection(line):
            dropped.append(line)
        else:
            kept.append(line)
    return "\n".join(kept).strip(), dropped
