#!/usr/bin/env python3
"""Evaluation script for translate terms feature using a list of user queries."""

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


def load_dataset(
    dataset_file: str = "benchmark/datasets/05_translate_terms_dataset.json", language: str = "en"
) -> list:
    """Load queries from dataset JSON file."""

    with open(dataset_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if language == "vi":
        return data.get("queries_vi", [])
    else:
        return data.get("queries_en", [])


async def run_single_query(query: str, workflow_service: WorkflowService, language: str = "en") -> dict:
    """Run workflow for a single query and capture results."""
    print(f"Running query: {query[:60]}...", end=" ", flush=True)

    start_time = datetime.now()
    result_data = None

    # Run workflow
    async for event in workflow_service.invoke_workflow(
        user_input=query,
        thread_id=f"eval-translate-{hash(query) % 10000}",
        top_k=10,
        stream=False,
    ):
        if event["type"] == "result":
            result_data = event.get("data", {})

    elapsed = (datetime.now() - start_time).total_seconds()

    response = format_response(result_data) if result_data else "Error"

    evaluation_result = {
        "query": query,
        "response": response,
        "execution_time": elapsed,
        "language": language,
    }

    # Always run LLM evaluation
    if result_data:
        llm_scores = await llm_evaluate_response(user_query=query, response=response, task=Task.TRANSLATION)
        evaluation_result["llm_scores"] = llm_scores
        print(
            f"✓ ({elapsed:.1f}s) [LLM: A={llm_scores.get('accuracy', '?')}"
            f"C={llm_scores.get('clarity', '?')}"
            f"L={llm_scores.get('language_correction', '?')}"
            f"]"
        )
    else:
        print(f"✓ ({elapsed:.1f}s)")

    return evaluation_result


async def run_evaluation(
    dataset_file: str, language: str = "en", output_file_name: str = "translate_terms_results_workflow.json"
):
    """Run evaluation on all user queries."""
    print("=" * 80)
    print("Medical Terms Translation Evaluation")
    print(f"Language: {language.upper()}")
    print("(with LLM-as-Judge automatic scoring)")
    print("=" * 80)

    # Load queries from dataset
    print(f"\nLoading queries from {dataset_file}...")
    user_queries = load_dataset(dataset_file, language=language)
    user_queries = user_queries[:2]
    if not user_queries:
        print(f"Error: No queries loaded from dataset for language '{language}'")
        return

    print(f"Found {len(user_queries)} queries to evaluate\n")

    # Initialize workflow service
    workflow_service = WorkflowService()

    # Run evaluation for each query with WorkflowService
    print("Running WorkflowService evaluation...")
    workflow_results = []
    for i, query in enumerate(user_queries, 1):
        print(f"[{i}/{len(user_queries)}] ", end="")
        result = await run_single_query(query, workflow_service, language=language)
        workflow_results.append(result)

    # Calculate average scores for WorkflowService
    workflow_llm_scored = [r for r in workflow_results if r.get("llm_scores", {}).get("accuracy") is not None]
    workflow_average_scores = {}
    if workflow_llm_scored:
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
        "total_queries": len(user_queries),
        "results": workflow_results,
        "average_scores": workflow_average_scores,
    }

    with open(workflow_output_file, "w", encoding="utf-8") as f:
        json.dump(workflow_output_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("✓ Evaluation complete!")
    print(f"WorkflowService results saved to: {workflow_output_file}")
    print(f"Total queries: {len(user_queries)}")
    print("=" * 80 + "\n")

    return str(workflow_output_file)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Medical Terms Translation Evaluation Script",
    )

    parser.add_argument(
        "--lang",
        "-l",
        choices=["en", "vi"],
        default="en",
        help="Language for evaluation: 'en' for English, 'vi' for Vietnamese",
    )
    args = parser.parse_args()

    dataset_file = "benchmark/datasets/05_translate_terms_dataset.json"
    # Evaluation mode (always uses LLM judge)
    output_file_name = f"translate_terms_results_workflow_{args.lang}_tmp.json"
    asyncio.run(run_evaluation(dataset_file=dataset_file, language=args.lang, output_file_name=output_file_name))


if __name__ == "__main__":
    main()
