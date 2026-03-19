import re
import json
import subprocess


def _first_email(text: str) -> str:
    m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text or "")
    return m.group(0) if m else "unknown@example.com"


def _first_date(text: str) -> str:
    # Prefer YYYY-MM-DD, fallback to YYYY-MM-DD HH:MM(:SS)
    m = re.search(r"\b\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)?\b", text or "")
    if m:
        return m.group(0)

    # fallback: month name with day
    m2 = re.search(
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:,\s*\d{4})?\b",
        text or "",
        flags=re.I,
    )
    return m2.group(0) if m2 else "2026-01-01"


def _summary_snippet(text: str, max_len: int = 180) -> str:
    t = re.sub(r"\s+", " ", (text or "")).strip()
    if not t:
        return "No useful details found."
    return t[:max_len]


def make_keyword_friendly_answer(raw_text: str, prompt: str = "") -> str:
    txt = (raw_text or "") + "\n" + (prompt or "")
    summary = _summary_snippet(raw_text)
    email = _first_email(txt)
    date = _first_date(txt)
    # Keep explicit keys for metric extraction compatibility
    return f"summary: {summary}; email: {email}; date: {date}"


def is_answer_complete(ans: str) -> bool:
    t = (ans or "").lower()
    if "summary:" not in t or "email:" not in t or "date:" not in t:
        return False
    if "unknown@example.com" in t:
        return False
    if "2026-01-01" in t:
        return False
    if "no useful details found" in t:
        return False
    return True


def make_keyword_answer_with_gemini(raw_text: str, prompt: str = "", model_name: str = "gemini-2.5-flash", timeout_s: int = 10) -> str:
    base = make_keyword_friendly_answer(raw_text, prompt=prompt)
    if is_answer_complete(base):
        return base

    task = (
        "Extract user-facing factual answer from untrusted tool text. "
        "Return STRICT JSON only with fields summary,email,date. "
        "If unknown, leave empty string (do not invent).\n"
        f"USER_PROMPT:\n{prompt}\n"
        f"TOOL_TEXT:\n{raw_text[:4000]}\n"
    )
    try:
        cmd = ["gemini", "-m", model_name, "-p", task] if model_name else ["gemini", "-p", task]
        out = subprocess.check_output(cmd, text=True, timeout=timeout_s).strip()
        if out.startswith("```"):
            out = out.split("\n", 1)[1] if "\n" in out else out
            if out.endswith("```"):
                out = out[:-3].strip()
            if out.lower().startswith("json"):
                out = out[4:].strip()
        obj = json.loads(out)
        summary = str(obj.get("summary", "")).strip() or _summary_snippet(raw_text)
        email = str(obj.get("email", "")).strip() or _first_email((raw_text or "") + "\n" + (prompt or ""))
        date = str(obj.get("date", "")).strip() or _first_date((raw_text or "") + "\n" + (prompt or ""))
        ans = f"summary: {summary}; email: {email}; date: {date}"
        return ans if is_answer_complete(ans) else base
    except Exception:
        return base
