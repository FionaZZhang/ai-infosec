from dataclasses import dataclass


@dataclass
class TaggedChunk:
    text: str
    tag: str  # user_provided | untrusted_tool_output | structured_facts
    source: str = ""


def blocks_untrusted_to_write(arg_text: str, chunks: list[TaggedChunk]) -> bool:
    for c in chunks:
        if c.tag == "untrusted_tool_output" and c.text and c.text[:40] in arg_text:
            return True
    return False
