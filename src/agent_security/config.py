from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class RunConfig:
    mode: str
    subset: str = "small"
    attack: str = "on"
    max_tasks: int = 30
    max_steps: int = 6
    seed: int = 42
    out_dir: str = "results"
    model_backend: str = os.getenv("MODEL_BACKEND", "heuristic")  # heuristic|gemini (summarizer backend)
    model_name: str = os.getenv("MODEL_NAME", "gemini-2.0-flash")
    policy_backend: str = os.getenv("POLICY_BACKEND", "heuristic")  # heuristic|gemini (interception backend)
    policy_threshold: float = float(os.getenv("POLICY_THRESHOLD", "0.6"))
    temperature: float = float(os.getenv("MODEL_TEMPERATURE", "0.0"))


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "src" / "agent_security" / "data"
DEFAULT_SUBSET_PATH = DATA_DIR / "agentdojo_small_subset.json"
SUBSET_50_PATH = DATA_DIR / "agentdojo_subset_50.json"
SUBSET_50_MIXED_PATH = DATA_DIR / "agentdojo_subset_50_mixed_official.json"
SUBSET_50_MULTIATTACK_PATH = DATA_DIR / "agentdojo_subset_50_multiattack.json"


def subset_path(name: str) -> Path:
    key = (name or "small").lower()
    if key in {"small", "tiny", "demo"}:
        return DEFAULT_SUBSET_PATH
    if key in {"50", "subset50", "small50", "mvp50"}:
        return SUBSET_50_PATH
    if key in {"subset50mix", "50mix", "official50", "mixed50"}:
        return SUBSET_50_MIXED_PATH
    if key in {"subset50multi", "multi50", "official50multi", "50multi"}:
        return SUBSET_50_MULTIATTACK_PATH
    return Path(name)
