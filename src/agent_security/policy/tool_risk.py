HIGH_RISK_TOOLS = {
    "send_email",
    "share_doc",
    "transfer_money",
    "delete_email",
    "post_message",
}

READ_TOOLS = {
    "read_email",
    "search_web",
    "browse_page",
    "list_docs",
}


def risk_of(tool_name: str) -> str:
    if tool_name in HIGH_RISK_TOOLS:
        return "HIGH"
    return "LOW"
