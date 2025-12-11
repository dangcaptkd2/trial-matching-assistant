"""Prompt loader utilities."""

from pathlib import Path


def _load_prompt(filename: str) -> str:
    """Load prompt text from template files."""
    templates_dir = Path(__file__).parent / "templates"
    path = templates_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8").strip()


intent_classification_prompt = _load_prompt("01_intent_classification_prompt.txt")
rerank_prompt = _load_prompt("02_rerank_prompt.txt")
synthesis_prompt = _load_prompt("03_synthesize_matching_prompt.txt")
reception_prompt = _load_prompt("04_reception_prompt.txt")
summarize_trial_prompt = _load_prompt("05_summarize_trial_prompt.txt")
check_eligibility_prompt = _load_prompt("06_check_eligibility_prompt.txt")
explain_criteria_prompt = _load_prompt("07_explain_criteria_prompt.txt")
compare_trials_prompt = _load_prompt("08_compare_trials_prompt.txt")
translate_terms_prompt = _load_prompt("09_translate_terms_prompt.txt")
clarification_prompt = _load_prompt("10_clarification_prompt.txt")
