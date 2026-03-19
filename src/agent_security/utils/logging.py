from pathlib import Path
import json
from datetime import datetime


class AuditLogger:
    def __init__(self, out_path: str):
        self.path = Path(out_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: dict):
        event = {
            "ts": datetime.utcnow().isoformat() + "Z",
            **event,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
