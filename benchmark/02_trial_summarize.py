#!/usr/bin/env python3
"""Evaluation script for trial summarization feature using 50 trial IDs."""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from benchmark.utilis import Task, llm_evaluate_response
from src.api.services.workflow import WorkflowService


def format_response(result: dict) -> str:
    """Extract the response from workflow result."""
    if result.get("chitchat_response"):
        return result.get("chitchat_response", "")
    elif result.get("final_answer"):
        return result.get("final_answer", "")
    else:
        return "No response generated."


def load_dataset(dataset_file) -> list:
    """Load trial IDs from dataset JSON file."""
    with open(dataset_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("trial_ids", [])


async def run_single_trial(trial_id: str, workflow_service: WorkflowService, language: str = "en") -> dict:
    """Run workflow for a single trial ID and capture results."""
    # Format query to trigger summarization based on language
    if language == "vi":
        query = f"Tóm tắt {trial_id}"
    else:
        query = f"Summarize {trial_id}"

    print(f"Running trial {trial_id} ({language})...", end=" ", flush=True)

    start_time = datetime.now()
    result_data = None

    async for event in workflow_service.invoke_workflow(
        user_input=query,
        thread_id=f"eval-summarize-{trial_id}",
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
        "execution_time": elapsed,
        "language": language,
    }

    # Always run LLM evaluation
    if result_data:
        llm_scores = await llm_evaluate_response(
            task=Task.SUMMARIZATION, trial_id=trial_id, response=response, user_input=query
        )
        evaluation_result["llm_scores"] = llm_scores
        print(
            f"✓ ({elapsed:.1f}s) [LLM: H={llm_scores.get('hallucination', '?')}"
            f"A={llm_scores.get('accuracy', '?')}"
            f"C={llm_scores.get('clarity', '?')}"
            f"L={llm_scores.get('language_correction', '?')}"
            f"]"
        )
    else:
        print(f"✓ ({elapsed:.1f}s)")

    return evaluation_result


async def run_evaluation(
    dataset_file: str, language: str = "en", output_file_name: str = "summarize_results_workflow.json"
):
    """Run evaluation on all trial IDs."""
    print("=" * 80)
    print("Clinical Trial Summarization Evaluation")
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

    # Initialize workflow service
    workflow_service = WorkflowService()

    # Run evaluation for each trial with WorkflowService
    print("Running WorkflowService evaluation...")
    workflow_results = []
    for i, trial_id in enumerate(trial_ids, 1):
        print(f"[{i}/{len(trial_ids)}] ", end="")
        result = await run_single_trial(trial_id, workflow_service, language=language)
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
        description="Clinical Trial Summarization Evaluation Script",
    )

    parser.add_argument(
        "--lang",
        "-l",
        choices=["en", "vi"],
        default="en",
        help="Language for evaluation: 'en' for English, 'vi' for Vietnamese",
    )

    args = parser.parse_args()

    dataset_file = "benchmark/datasets/02_summarize_dataset.json"
    # Evaluation mode (always uses LLM judge)
    output_file_name = f"summarize_results_workflow_{args.lang}_tmp.json"
    asyncio.run(run_evaluation(dataset_file=dataset_file, language=args.lang, output_file_name=output_file_name))


if __name__ == "__main__":
    main()
