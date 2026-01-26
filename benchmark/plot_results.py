import os

import matplotlib.pyplot as plt
import numpy as np

# Create directory if it doesn't exist
output_dir = "benchmark/results/plots"
os.makedirs(output_dir, exist_ok=True)

# Data
systems = ["ChatGPT", "Workflow (Nano)", "Workflow (MedGemma)"]
colors = ["#FF9999", "#66B2FF", "#99FF99"]

# 1. Quality Metrics (Score 0-5)
metrics = ["Depth", "Relevance", "Clarity", "Completeness"]

# English Data
eng_scores = {
    "ChatGPT": [2.2, 3.1, 5.0, 2.4],
    "Workflow (Nano)": [2.40, 3.00, 4.60, 2.30],
    "Workflow (MedGemma)": [2.00, 2.20, 4.10, 1.90],
}

# Vietnamese Data
vie_scores = {
    "ChatGPT": [1.1, 1.5, 4.3, 1.2],
    "Workflow (Nano)": [2.50, 3.20, 4.50, 2.40],
    "Workflow (MedGemma)": [2.10, 2.00, 4.10, 1.90],
}


def plot_grouped_bar(scores_data, title, filename):
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, system in enumerate(systems):
        offset = (i - 1) * width
        ax.bar(x + offset, scores_data[system], width, label=system, color=colors[i], edgecolor="grey")

    ax.set_ylabel("Score (0-5)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 5.5)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


plot_grouped_bar(eng_scores, "Quality Metrics Comparison (English)", "english_quality_metrics.png")
plot_grouped_bar(vie_scores, "Quality Metrics Comparison (Vietnamese)", "vietnamese_quality_metrics.png")

# 2. Retrieval Performance (Trial IDs found)
# English: 0.2, 2.70, 4.10
# Vietnamese: 0.0, 3.00, 4.10
eng_ids = [0.2, 2.70, 4.10]
vie_ids = [0.0, 3.00, 4.10]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(systems))
width = 0.35

ax.bar(x - width / 2, eng_ids, width, label="English", color="#FFCC99", edgecolor="grey")
ax.bar(x + width / 2, vie_ids, width, label="Vietnamese", color="#99CCFF", edgecolor="grey")

ax.set_ylabel("Number of Trial IDs Found")
ax.set_title("Retrieval Performance (Trial IDs)")
ax.set_xticks(x)
ax.set_xticklabels(systems)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.7)

for i, v in enumerate(eng_ids):
    ax.text(i - width / 2, v + 0.1, str(v), ha="center", va="bottom")
for i, v in enumerate(vie_ids):
    ax.text(i + width / 2, v + 0.1, str(v), ha="center", va="bottom")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "retrieval_performance.png"))
plt.close()

# 3. Execution Time
# English: 4.3, 4.68, 14.87
# Vietnamese: 4.8, 10.16, 13.32
eng_time = [4.3, 4.68, 14.87]
vie_time = [4.8, 10.16, 13.32]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(systems))

ax.bar(x - width / 2, eng_time, width, label="English", color="#FFCC99", edgecolor="grey")
ax.bar(x + width / 2, vie_time, width, label="Vietnamese", color="#99CCFF", edgecolor="grey")

ax.set_ylabel("Time (seconds)")
ax.set_title("Execution Time Comparison")
ax.set_xticks(x)
ax.set_xticklabels(systems)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.7)

for i, v in enumerate(eng_time):
    ax.text(i - width / 2, v + 0.2, str(v), ha="center", va="bottom")
for i, v in enumerate(vie_time):
    ax.text(i + width / 2, v + 0.2, str(v), ha="center", va="bottom")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "execution_time.png"))
plt.close()

# 4. Token Usage & Cost (Total) - English Only for simplicity? Or Combined?
# Let's do Cost only, sum of English + Vietnamese maybe? Or just average?
# The report shows cost per language test.
# Let's plot Cost for English and Vietnamese side by side.
# English Cost: 0.004034, 0.004334, 0.000000
# Vietnamese Cost: 0.004215, 0.005142, 0.000000
eng_cost = [0.004034, 0.004334, 0.0]
vie_cost = [0.004215, 0.005142, 0.0]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(systems))

ax.bar(x - width / 2, eng_cost, width, label="English", color="#FFCC99", edgecolor="grey")
ax.bar(x + width / 2, vie_cost, width, label="Vietnamese", color="#99CCFF", edgecolor="grey")

ax.set_ylabel("Cost (USD)")
ax.set_title("Cost Estimate Comparison")
ax.set_xticks(x)
ax.set_xticklabels(systems)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.7)

for i, v in enumerate(eng_cost):
    if v > 0:
        ax.text(i - width / 2, v + 0.0001, f"${v:.4f}", ha="center", va="bottom", fontsize=9)
for i, v in enumerate(vie_cost):
    if v > 0:
        ax.text(i + width / 2, v + 0.0001, f"${v:.4f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "cost_comparison.png"))
plt.close()

print(f"Plots saved to {output_dir}")
