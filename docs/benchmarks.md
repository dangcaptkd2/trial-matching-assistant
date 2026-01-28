# Benchmarking & Evaluation

The project includes a comprehensive benchmarking suite to evaluate the performance and accuracy of the Clinical Trial Assistant's various capabilities. These scripts are located in the `benchmark/` directory.

## Evaluation Areas

The benchmark suite covers the following core capabilities:

| Area | Script | Description |
|------|--------|-------------|
| **Trial Matching** | `01_trial_matching.py` | Evaluates the system's ability to find relevant trials for a given patient profile. |
| **Summarization** | `02_trial_summarize.py` | Tests the quality of generated clinical trial summaries against ground truth or heuristics. |
| **Eligibility Check** | `03_trial_eligibility.py` | Assesses the accuracy of determining if a patient meets specific inclusion/exclusion criteria. |
| **Criterion Explanation**| `04_trial_explain_criteria.py` |  Evaluates how well the agent explains complex medical criteria in plain language. |
| **Term Translation** | `05_trial_translate_terms.py` | Tests the translation of medical jargon into understandable terms. |
| **Trial Comparison** | `06_trial_compare.py` | evaluating the agent's ability to compare multiple trials side-by-side. |

## Methodology

### 1. Dataset Generation
- Scripts like `create_datasets.py` are likely used to generate or format the test datasets (`ids`, `queries`, `ground_truth`) located in `benchmark/datasets/`.
- Synthetic patient profiles or real anonymized queries may be used.

### 2. Execution
Each benchmark script typically:
- Loads a dataset of test cases.
- Runs the specific module of the Agent (e.g., calling the `search` node or `summarize` function).
- Captures the output (Agent's response).
- Compares the output against a Ground Truth or uses an LLM-as-a-Judge approach to score the quality.

### 3. Metrics
Common metrics likely collected include:
- **Precision/Recall**: For trial matching (did we find the right trials?).
- **Latency**: Time taken to generate a response.
- **Factuality**: (For summarization) Does the summary hallucinate?
- **Readability**: (For explanation) Is the explanation simple enough?

### 4. Reporting
- `generate_report.py`: Aggregates the results from individual runs.
- `plot_results.py`: Visualizes the metrics (e.g., charts showing performance data).
- Results are stored in the `benchmark/results/` directory for historical tracking.

## Running Benchmarks

To run a specific benchmark, execute the corresponding script from the root directory:

```bash
# Run Trial Matching Benchmark
python benchmark/01_trial_matching.py

# Run Eligibility Check Benchmark
python benchmark/03_trial_eligibility.py
```

## Comparisons

The system also includes scripts for comparing against competitors or baselines:
- `competitor_agent.py`: Implementation of a baseline or competitor logic.
- `01_trial_matching_comparison.py`: Side-by-side comparison of the matchmaking algorithm.
