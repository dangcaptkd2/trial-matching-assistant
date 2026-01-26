import json
from pathlib import Path

# Configuration for file paths
RESULTS_DIR = Path(__file__).parent / "results"
MODELS = ["gpt-4.1-nano", "medgemma-4b-it"]
LANGUAGES = ["en", "vi"]

# Cost Configuration (USD per 1M tokens)
# Adjust these values based on actual pricing
COST_CONFIG = {
    "gpt-4.1-nano": {
        "input": 0.15,  # Example: $0.15 / 1M tokens (GPT-4o-mini approx)
        "output": 0.60,  # Example: $0.60 / 1M tokens
    },
    "medgemma-4b-it": {
        "input": 0.0,  # Set to 0 if self-hosted or unknown
        "output": 0.0,
    },
}

LABELS_MAP = {
    "trial_id_count": "Số lượng Trial ID",
    "execution_time_avg": "Thời gian thực thi (giây)",
    "depth": "Độ sâu (Depth)",
    "relevance": "Mức độ liên quan (Relevance)",
    "clarity": "Độ rõ ràng (Clarity)",
    "completeness": "Độ đầy đủ (Completeness)",
    # Token metrics labels
    "avg_prompt_tokens": "Avg Prompt Tokens",
    "avg_completion_tokens": "Avg Completion Tokens",
    "avg_total_tokens": "Avg Total Tokens",
    "total_prompt_tokens": "Total Prompt Tokens",
    "total_completion_tokens": "Total Completion Tokens",
    "total_tokens": "Total Tokens",
}

ORDERED_METRICS = ["trial_id_count", "execution_time_avg", "depth", "relevance", "clarity", "completeness"]

TOKEN_METRICS = [
    "avg_prompt_tokens",
    "avg_completion_tokens",
    "avg_total_tokens",
    "total_prompt_tokens",
    "total_completion_tokens",
    "total_tokens",
]


def load_data(model, lang):
    filename = f"01_matching_comparison_results_{lang}.json"
    file_path = RESULTS_DIR / model / filename
    if not file_path.exists():
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_cost(model, prompt_tokens, completion_tokens):
    prices = COST_CONFIG.get(model, {"input": 0, "output": 0})
    input_cost = (prompt_tokens / 1_000_000) * prices["input"]
    output_cost = (completion_tokens / 1_000_000) * prices["output"]
    return input_cost + output_cost


def generate_markdown_table(lang):
    # Prepare header
    header = ["Tiêu chí", "ChatGPT"] + [f"Workflow ({m})" for m in MODELS]

    # Collect data columns
    chatgpt_data = None
    data_columns = {}

    # Load ChatGPT data (try from the first model)
    first_model_data = load_data(MODELS[0], lang)
    if first_model_data and "average_scores" in first_model_data and "chatgpt" in first_model_data["average_scores"]:
        chatgpt_data = first_model_data["average_scores"]["chatgpt"]

    # Load Workflow data for all models
    for model in MODELS:
        data = load_data(model, lang)
        if data and "average_scores" in data and "workflow" in data["average_scores"]:
            data_columns[model] = data["average_scores"]["workflow"]
        else:
            data_columns[model] = None

    rows = []

    # Standard Metrics
    for metric_key in ORDERED_METRICS:
        row = [LABELS_MAP.get(metric_key, metric_key)]

        # ChatGPT column
        if chatgpt_data:
            val = chatgpt_data.get(metric_key, "N/A")
            if isinstance(val, float):
                val = f"{val:.1f}"  # User image shows 1 decimal place usually, but let's stick to simple formatting
            row.append(str(val))
        else:
            row.append("N/A")

        # Workflow columns
        for model in MODELS:
            stats = data_columns[model]
            if stats:
                val = stats.get(metric_key, "N/A")
                if isinstance(val, float):
                    val = f"{val:.2f}"
                row.append(str(val))
            else:
                row.append("N/A")
        rows.append(row)

    # Token Metrics Header
    rows.append(["**Token Usage**", ""] + [""] * len(MODELS))

    for metric_key in TOKEN_METRICS:
        row = [LABELS_MAP.get(metric_key, metric_key)]

        # ChatGPT column
        if chatgpt_data:
            val = chatgpt_data.get(metric_key, "N/A")
            if isinstance(val, (int, float)):
                if "avg" in metric_key:
                    val = f"{val:.1f}"
                else:
                    val = int(val)
            row.append(str(val))
        else:
            row.append("N/A")

        # Workflow columns
        for model in MODELS:
            stats = data_columns[model]
            if stats:
                val = stats.get(metric_key, "N/A")
                if isinstance(val, (int, float)):
                    if "avg" in metric_key:
                        val = f"{val:.1f}"
                    else:
                        val = int(val)
                row.append(str(val))
            else:
                row.append("N/A")
        rows.append(row)

    # Cost Estimation
    rows.append(["**Cost Estimate (Total)**", ""] + [""] * len(MODELS))
    row_cost = ["Total Cost (USD)"]

    # ChatGPT Cost
    if chatgpt_data:
        # Assuming ChatGPT uses same pricing as gpt-4.1-nano for now, or we could add specific config
        # Using gpt-4.1-nano price as proxy for "ChatGPT" baseline cost if essentially same model class
        p_tokens = chatgpt_data.get("total_prompt_tokens", 0)
        c_tokens = chatgpt_data.get("total_completion_tokens", 0)
        # We can use gpt-4.1-nano pricing for ChatGPT baseline
        cost = calculate_cost("gpt-4.1-nano", p_tokens, c_tokens)
        row_cost.append(f"${cost:.6f}")
    else:
        row_cost.append("N/A")

    # Workflow Costs
    for model in MODELS:
        stats = data_columns[model]
        if stats:
            p_tokens = stats.get("total_prompt_tokens", 0)
            c_tokens = stats.get("total_completion_tokens", 0)
            cost = calculate_cost(model, p_tokens, c_tokens)
            row_cost.append(f"${cost:.6f}")
        else:
            row_cost.append("N/A")
    rows.append(row_cost)

    # Construct Markdown
    md = f"| {' | '.join(header)} |\n"
    md += f"| {' | '.join(['---'] * len(header))} |\n"
    for row in rows:
        md += f"| {' | '.join(row)} |\n"

    return md


def main():
    report = "# Đánh giá chất lượng\n\n"

    # Tiếng Anh
    report += "## Tiếng Anh\n\n"
    report += generate_markdown_table("en")
    report += "\n\n"

    # Tiếng Việt
    report += "## Tiếng Việt\n\n"
    report += generate_markdown_table("vi")

    print("\nGenerated Report:\n")
    print(report)

    # Save to file
    output_path = RESULTS_DIR / "comparison_report.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
