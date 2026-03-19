import json
from pathlib import Path
from typing import Any, Union


def dump_json(path: Union[str, Path], obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def load_json(path: Union[str, Path]) -> Any:
    return json.loads(Path(path).read_text())


def dump_jsonl(path: Union[str, Path], rows: list) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
