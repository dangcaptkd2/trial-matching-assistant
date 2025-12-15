#!/usr/bin/env python3
"""Evaluation script for eligibility checking feature using topics2023.xml and qrels2022.txt."""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from benchmark.utilis import Task, llm_evaluate_response
from src.api.services.workflow import WorkflowService
from src.config.settings import settings
from src.services.search import ElasticsearchTrialSearcher


def format_response(result: dict) -> str:
    """Extract the response from workflow result."""
    if result.get("chitchat_response"):
        return result.get("chitchat_response", "")
    elif result.get("final_answer"):
        return result.get("final_answer", "")
    else:
        return "No response generated."


def get_ground_truth_label_info(label: int) -> tuple[str, str]:
    """Get ground truth label name and explanation."""
    label_map = {
        0: ("Not Relevant", "The trial is not relevant to the patient's condition or situation."),
        1: ("Excluded", "The patient is explicitly excluded from the trial based on eligibility criteria."),
        2: ("Eligible", "The patient meets the eligibility criteria and is eligible for the trial."),
    }
    return label_map.get(label, ("Unknown", "Unknown label"))


async def fetch_trial_data(trial_id: str, es_searcher: ElasticsearchTrialSearcher) -> dict:
    """Fetch trial data from Elasticsearch."""

    doc = es_searcher.es.client.get(index=es_searcher.index_name, id=trial_id)
    source = doc.get("_source", {})
    return {
        "nct_id": trial_id,
        "title": source.get("brief_title") or source.get("official_title", "N/A"),
        "eligibility_criteria": source.get("eligibility_criteria", ""),
        "brief_summary": source.get("brief_summary", ""),
        "detailed_description": source.get("detailed_description", ""),
    }


def load_dataset(dataset_file: str = "benchmark/datasets/03_eligibility_dataset.json") -> list:
    """Load samples from dataset JSON file."""
    with open(dataset_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("samples", [])


async def run_single_eligibility_check(
    topic_number: str,
    trial_id: str,
    ground_truth_label: int,
    patient_profile: str,
    workflow_service: WorkflowService,
    es_searcher: ElasticsearchTrialSearcher,
    language: str = "en",
) -> dict:
    """Run workflow for a single topic-trial pair and capture results."""
    # Format query to trigger eligibility check based on language
    if language == "vi":
        query = f"Bệnh nhân này có đủ điều kiện cho {trial_id} không? {patient_profile}"
    else:
        query = f"Is this patient eligible for {trial_id}? {patient_profile}"

    print(
        f"Running topic {topic_number} + trial {trial_id} (label={ground_truth_label}, {language})...",
        end=" ",
        flush=True,
    )

    start_time = datetime.now()
    result_data = None

    # Fetch trial data for LLM judge
    trial_data = await fetch_trial_data(trial_id, es_searcher)

    # Run workflow
    async for event in workflow_service.invoke_workflow(
        user_input=query,
        thread_id=f"eval-eligibility-{topic_number}-{trial_id}",
        top_k=10,
        stream=False,
    ):
        if event["type"] == "result":
            result_data = event.get("data", {})

    elapsed = (datetime.now() - start_time).total_seconds()

    response = format_response(result_data) if result_data else "Error"

    evaluation_result = {
        "topic_number": topic_number,
        "trial_id": trial_id,
        "ground_truth_label": ground_truth_label,
        "patient_profile": patient_profile,
        "query": query,
        "response": response,
        "trial_data": trial_data,
        "execution_time": elapsed,
        "language": language,
    }

    # Always run LLM evaluation
    if result_data and "error" not in trial_data:
        label_name, label_explanation = get_ground_truth_label_info(ground_truth_label)

        # Format trial information
        trial_id = trial_data.get("nct_id", "N/A")
        trial_title = trial_data.get("title", "N/A")
        eligibility_criteria = trial_data.get("eligibility_criteria", "Not available")

        llm_scores = await llm_evaluate_response(
            task=Task.ELIGIBILITY,
            user_input=query,
            patient_profile=patient_profile,
            trial_id=trial_id,
            trial_title=trial_title,
            eligibility_criteria=eligibility_criteria,
            ground_truth_label=str(ground_truth_label),
            label_name=label_name,
            workflow_answer=response,
            ground_truth_explanation=label_explanation,
        )
        evaluation_result["llm_scores"] = llm_scores
        print(
            f"✓ ({elapsed:.1f}s) [LLM: H={llm_scores.get('hallucination', '?')} A={llm_scores.get('accuracy', '?')} C={llm_scores.get('clarity', '?')} L={llm_scores.get('language_correction', '?')}]"
        )
    else:
        print(f"✓ ({elapsed:.1f}s)")

    return evaluation_result


def calculate_average_scores(results: list) -> dict:
    """Calculate average scores from results."""
    llm_scored = [r for r in results if r.get("llm_scores", {}).get("hallucination") is not None]

    average_scores = {}
    if llm_scored:
        avg_h = sum(r["llm_scores"]["hallucination"] for r in llm_scored) / len(llm_scored)
        avg_a = sum(r["llm_scores"]["accuracy"] for r in llm_scored) / len(llm_scored)
        avg_c = sum(r["llm_scores"]["clarity"] for r in llm_scored) / len(llm_scored)
        lang_corr_scored = [r for r in llm_scored if r["llm_scores"].get("language_correction") is not None]
        avg_l = (
            sum(r["llm_scores"]["language_correction"] for r in lang_corr_scored) / len(lang_corr_scored)
            if lang_corr_scored
            else None
        )
        avg_latency = sum(r.get("execution_time", 0) for r in results) / len(results)

        average_scores = {
            "overall": {
                "hallucination": round(avg_h, 2),
                "accuracy": round(avg_a, 2),
                "clarity": round(avg_c, 2),
                "language_correction": round(avg_l, 2) if avg_l is not None else None,
                "latency": round(avg_latency, 2),
                "total_scored": len(llm_scored),
            }
        }

        # Calculate averages by ground truth label
        by_label = {}
        for label in [0, 1, 2]:
            label_results = [r for r in llm_scored if r.get("ground_truth_label") == label]
            if label_results:
                label_name, _ = get_ground_truth_label_info(label)
                avg_h_label = sum(r["llm_scores"]["hallucination"] for r in label_results) / len(label_results)
                avg_a_label = sum(r["llm_scores"]["accuracy"] for r in label_results) / len(label_results)
                avg_c_label = sum(r["llm_scores"]["clarity"] for r in label_results) / len(label_results)
                lang_corr_label = [r for r in label_results if r["llm_scores"].get("language_correction") is not None]
                avg_l_label = (
                    sum(r["llm_scores"]["language_correction"] for r in lang_corr_label) / len(lang_corr_label)
                    if lang_corr_label
                    else None
                )

                by_label[str(label)] = {
                    "label_name": label_name,
                    "hallucination": round(avg_h_label, 2),
                    "accuracy": round(avg_a_label, 2),
                    "clarity": round(avg_c_label, 2),
                    "language_correction": round(avg_l_label, 2) if avg_l_label is not None else None,
                    "count": len(label_results),
                }

        if by_label:
            average_scores["by_label"] = by_label

    return average_scores


async def run_evaluation(
    dataset_file: str,
    output_file_name: str,
    language: str = "en",
):
    """Run evaluation on sampled topic-trial pairs."""
    print("=" * 80)
    print("Clinical Trial Eligibility Evaluation")
    print(f"Language: {language.upper()}")
    print("(with LLM-as-Judge automatic scoring)")
    print("=" * 80)

    # Load samples from dataset
    print(f"\nLoading samples from {dataset_file}...")
    samples = load_dataset(dataset_file)
    samples = samples[:2]
    if not samples:
        print("Error: No samples loaded from dataset")
        return

    print(f"Found {len(samples)} samples to evaluate\n")

    # Initialize services
    workflow_service = WorkflowService()
    es_searcher = ElasticsearchTrialSearcher(index_name=settings.es_index_name)

    # Run evaluation for each sample with WorkflowService
    print("Running WorkflowService evaluation...")
    workflow_results = []
    for i, sample in enumerate(samples, 1):
        topic_number = sample["topic_number"]
        trial_id = sample["trial_id"]
        label = sample["ground_truth_label"]
        patient_profile = sample["patient_profile"]

        print(f"\n[{i}/{len(samples)}] ", end="")
        result = await run_single_eligibility_check(
            topic_number,
            trial_id,
            label,
            patient_profile,
            workflow_service,
            es_searcher,
            language=language,
        )
        workflow_results.append(result)

    # Calculate average scores
    workflow_average_scores = calculate_average_scores(workflow_results)

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
        "total_samples": len(samples),
        "results": workflow_results,
        "average_scores": workflow_average_scores,
    }

    with open(workflow_output_file, "w", encoding="utf-8") as f:
        json.dump(workflow_output_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("✓ Evaluation complete!")
    print(f"WorkflowService results saved to: {workflow_output_file}")
    print(f"Total samples: {len(samples)}")
    print("=" * 80 + "\n")

    return str(workflow_output_file)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Clinical Trial Eligibility Evaluation Script",
    )

    parser.add_argument(
        "--lang",
        "-l",
        choices=["en", "vi"],
        default="en",
        help="Language for evaluation: 'en' for English, 'vi' for Vietnamese",
    )

    args = parser.parse_args()

    dataset_file = "benchmark/datasets/03_eligibility_dataset.json"
    output_file_name = f"eligibility_results_workflow_{args.lang}.json"
    # Evaluation mode (always uses LLM judge)
    asyncio.run(run_evaluation(dataset_file=dataset_file, language=args.lang, output_file_name=output_file_name))


if __name__ == "__main__":
    main()
