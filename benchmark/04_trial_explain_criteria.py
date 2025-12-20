#!/usr/bin/env python3
"""Evaluation script for explain criteria feature using 50 random trial IDs from Elasticsearch."""

import argparse
import asyncio
import json
import random
from datetime import datetime
from pathlib import Path

from benchmark.utilis import Task, llm_evaluate_response
from src.api.services.workflow import WorkflowService
from src.config.settings import settings
from src.services.es_search import ElasticsearchTrialSearcher


def format_response(result: dict) -> str:
    """Extract the response from workflow result."""
    if result.get("chitchat_response"):
        return result.get("chitchat_response", "")
    elif result.get("final_answer"):
        return result.get("final_answer", "")
    else:
        return "No response generated."


async def get_random_trial_ids(es_searcher: ElasticsearchTrialSearcher, count: int = 50, seed: int = 42) -> list[str]:
    """Get random trial IDs from Elasticsearch."""
    if seed is not None:
        random.seed(seed)

    # Use match_all query to get a large sample, then randomly select
    body = {
        "size": min(count * 3, 10000),  # Get more than needed, but cap at 10k
        "query": {"match_all": {}},
        "_source": ["nct_id"],
    }

    def _do_search():
        return es_searcher.es.client.search(index=es_searcher.index_name, body=body)

    resp = await asyncio.get_running_loop().run_in_executor(None, _do_search)
    hits = resp.get("hits", {}).get("hits", [])

    # Extract trial IDs
    trial_ids = []
    for hit in hits:
        nct_id = hit.get("_id") or hit.get("_source", {}).get("nct_id")
        if nct_id:
            trial_ids.append(nct_id)

    # Randomly sample the requested count
    if len(trial_ids) < count:
        print(f"Warning: Only {len(trial_ids)} trials found, requested {count}")
        return trial_ids

    return random.sample(trial_ids, count)


async def fetch_trial_data(trial_id: str, es_searcher: ElasticsearchTrialSearcher) -> dict:
    """Fetch trial data from Elasticsearch."""

    doc = es_searcher.es.client.get(index=es_searcher.index_name, id=trial_id)
    source = doc.get("_source", {})
    return {
        "nct_id": trial_id,
        "title": source.get("brief_title") or source.get("official_title", "N/A"),
        "eligibility_criteria": source.get("eligibility_criteria", ""),
    }


def load_dataset(dataset_file: str) -> list:
    """Load trial IDs from dataset JSON file."""

    with open(dataset_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("trial_ids", [])


async def run_single_trial(
    trial_id: str, workflow_service: WorkflowService, es_searcher: ElasticsearchTrialSearcher, language: str = "en"
) -> dict:
    """Run workflow for a single trial ID and capture results."""
    # Format query to trigger explain criteria based on language
    if language == "vi":
        query = f"Hãy giải thích các tiêu chí đủ điều kiện của thử nghiệm {trial_id}"
    else:
        query = f"Explain the eligibility criteria for trial {trial_id}"

    print(f"Running trial {trial_id}...", end=" ", flush=True)

    start_time = datetime.now()
    result_data = None

    # Fetch trial data for LLM judge
    trial_data = await fetch_trial_data(trial_id, es_searcher)

    # Run workflow
    async for event in workflow_service.invoke_workflow(
        user_input=query,
        thread_id=f"eval-explain-{trial_id}",
        top_k=10,
        stream=False,
    ):
        if event["type"] == "result":
            result_data = event.get("data", {})

    elapsed = (datetime.now() - start_time).total_seconds()

    response = format_response(result_data) if result_data else "Error"

    evaluation_result = {
        "trial_id": trial_id,
        "query": query,
        "response": response,
        "trial_data": trial_data,
        "execution_time": elapsed,
        "language": language,
    }

    # Always run LLM evaluation
    if result_data and "error" not in trial_data:
        llm_scores = await llm_evaluate_response(
            task=Task.EXPLANATION,
            user_input=query,
            trial_id=trial_id,
            trial_title=trial_data.get("title", "N/A"),
            eligibility_criteria=trial_data.get("eligibility_criteria", ""),
            response=response,
        )
        evaluation_result["llm_scores"] = llm_scores
        print(
            f"✓ ({elapsed:.1f}s) [LLM: H={llm_scores.get('hallucination', '?')} A={llm_scores.get('accuracy', '?')} C={llm_scores.get('clarity', '?')} L={llm_scores.get('language_correction', '?')}]"
        )
    else:
        print(f"✓ ({elapsed:.1f}s)")

    return evaluation_result


async def run_evaluation(
    dataset_file: str, language: str = "en", output_file_name: str = "explain_criteria_results_workflow.json"
):
    """Run evaluation on trial IDs from dataset."""
    print("=" * 80)
    print("Clinical Trial Criteria Explanation Evaluation")
    print(f"Language: {language.upper()}")
    print("(with LLM-as-Judge automatic scoring)")
    print("=" * 80)

    # Load trial IDs from dataset
    print(f"\nLoading trial IDs from {dataset_file}...")
    trial_ids = load_dataset(dataset_file)
    trial_ids = trial_ids[:2]
    if not trial_ids:
        print("Error: No trial IDs loaded from dataset")
        return

    print(f"Found {len(trial_ids)} trials to evaluate\n")

    # Initialize services
    es_searcher = ElasticsearchTrialSearcher(index_name=settings.es_index_name)
    workflow_service = WorkflowService()

    # Run evaluation for each trial with WorkflowService
    print("Running WorkflowService evaluation...")
    workflow_results = []
    for i, trial_id in enumerate(trial_ids, 1):
        print(f"[{i}/{len(trial_ids)}] ", end="")
        result = await run_single_trial(trial_id, workflow_service, es_searcher, language=language)
        workflow_results.append(result)

    # Calculate average scores for WorkflowService
    workflow_llm_scored = [r for r in workflow_results if r.get("llm_scores", {}).get("hallucination") is not None]
    workflow_average_scores = {}
    if workflow_llm_scored:
        avg_h = sum(r["llm_scores"]["hallucination"] for r in workflow_llm_scored) / len(workflow_llm_scored)
        avg_a = sum(r["llm_scores"]["accuracy"] for r in workflow_llm_scored) / len(workflow_llm_scored)
        avg_c = sum(r["llm_scores"]["clarity"] for r in workflow_llm_scored) / len(workflow_llm_scored)
        lang_corr_scored = [r for r in workflow_llm_scored if r["llm_scores"].get("language_correction") is not None]
        avg_l = (
            sum(r["llm_scores"]["language_correction"] for r in lang_corr_scored) / len(lang_corr_scored)
            if lang_corr_scored
            else None
        )
        avg_latency = sum(r.get("execution_time", 0) for r in workflow_results) / len(workflow_results)

        workflow_average_scores = {
            "overall": {
                "hallucination": round(avg_h, 2),
                "accuracy": round(avg_a, 2),
                "clarity": round(avg_c, 2),
                "language_correction": round(avg_l, 2) if avg_l is not None else None,
                "latency": round(avg_latency, 2),
                "total_scored": len(workflow_llm_scored),
            }
        }

    # Save results
    output_dir = Path("benchmark/results")
    output_dir.mkdir(exist_ok=True)

    # Save WorkflowService results
    workflow_output_file = output_dir / output_file_name
    workflow_output_data = {
        "timestamp": datetime.now().isoformat(),
        "dataset_file": dataset_file,
        "agent": "workflow",
        "language": language,
        "total_trials": len(trial_ids),
        "results": workflow_results,
        "average_scores": workflow_average_scores,
    }

    with open(workflow_output_file, "w", encoding="utf-8") as f:
        json.dump(workflow_output_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("✓ Evaluation complete!")
    print(f"WorkflowService results saved to: {workflow_output_file}")
    print(f"Total trials: {len(trial_ids)}")
    print("=" * 80 + "\n")

    return str(workflow_output_file)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Clinical Trial Criteria Explanation Evaluation Script",
    )

    parser.add_argument(
        "--lang",
        "-l",
        choices=["en", "vi"],
        default="en",
        help="Language for evaluation: 'en' for English, 'vi' for Vietnamese",
    )

    args = parser.parse_args()
    dataset_file = "benchmark/datasets/04_explain_criteria_dataset.json"
    # Evaluation mode (always uses LLM judge)
    output_file_name = f"explain_criteria_results_workflow_{args.lang}_tmp.json"
    asyncio.run(run_evaluation(dataset_file=dataset_file, language=args.lang, output_file_name=output_file_name))


if __name__ == "__main__":
    main()
