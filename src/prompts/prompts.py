"""Prompt loader utilities."""

from pathlib import Path


def _load_prompt(filename: str) -> str:
    """Load prompt text from template files."""
    templates_dir = Path(__file__).parent / "templates"
    path = templates_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8").strip()


intent_classification_prompt = _load_prompt("intent_classification_prompt.txt")
rerank_prompt = _load_prompt("rerank_prompt.txt")
synthesis_prompt = _load_prompt("synthesis_prompt.txt")
reception_prompt = _load_prompt("reception_prompt.txt")
summarize_trial_prompt = _load_prompt("summarize_trial_prompt.txt")
clarification_prompt = _load_prompt("clarification_prompt.txt")
