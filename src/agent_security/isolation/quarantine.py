from dataclasses import dataclass, asdict


@dataclass
class QuarantinedItem:
    tool_name: str
    raw_output: str
    source: str


class QuarantineBuffer:
    def __init__(self):
        self.items: list[QuarantinedItem] = []

    def add(self, tool_name: str, raw_output: str, source: str):
        self.items.append(QuarantinedItem(tool_name=tool_name, raw_output=raw_output, source=source))

    def dump(self):
        return [asdict(i) for i in self.items]
