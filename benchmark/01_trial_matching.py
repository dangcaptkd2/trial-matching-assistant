"""Evaluation script for patient matching feature using 50 TREC topics."""

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


def extract_trials_found(result: dict) -> list:
    """Extract trial information from result."""
    trials = []
    if result.get("reranked_results"):
        for trial in result.get("reranked_results", []):
            trials.append(
                {
                    "nct_id": trial.get("nct_id", "N/A"),
                    "title": trial.get("title", "N/A"),
                    "llm_match_score": trial.get("llm_match_score", 0.0),
                    "match_reasoning": trial.get("match_reasoning", "N/A"),
                }
            )
    return trials


async def run_single_topic(topic: dict, workflow_service: WorkflowService, language: str = "en") -> dict:
    """Run workflow for a single topic and capture results."""
    topic_number = topic["number"]

    # Get patient profile based on language
    if language == "vi":
        patient_profile = topic.get("vi_content", topic.get("content", ""))
        query = "tìm thử nghiệm lâm sàng cho bệnh nhân này: " + patient_profile
    else:
        patient_profile = topic.get("content", "")
        query = "find trials for this patient: " + patient_profile

    print(f"Running topic {topic_number} ({language})...", end=" ", flush=True)

    start_time = datetime.now()
    result_data = None

    async for event in workflow_service.invoke_workflow(
        user_input=query,
        thread_id=f"eval-topic-{topic_number}-{language}",
        top_k=10,
        stream=False,
    ):
        if event["type"] == "result":
            result_data = event.get("data", {})

    elapsed = (datetime.now() - start_time).total_seconds()

    response = format_response(result_data) if result_data else "Error"
    trials = extract_trials_found(result_data) if result_data else []

    evaluation_result = {
        "topic_number": topic_number,
        "patient_profile": patient_profile,
        "response": response,
        "trials_found": trials,
        "execution_time": elapsed,
        "language": language,
    }

    # Always run LLM evaluation
    if result_data:
        llm_scores = await llm_evaluate_response(
            task=Task.MATCHING,
            patient_profile=patient_profile,
            response=response,
            trials_found=trials,
            user_input=query,
        )
        evaluation_result["llm_scores"] = llm_scores
        print(
            f"✓ ({elapsed:.1f}s) [LLM: H={llm_scores.get('hallucination', '?')}"
            f"A={llm_scores.get('accuracy', '?')}"
            f"C={llm_scores.get('clarity', '?')}"
            f"L={llm_scores.get('language_correction', '?')}"
            f"]"
        )
        evaluation_result["llm_scores"] = llm_scores
    else:
        print(f"✓ ({elapsed:.1f}s)")

    return evaluation_result


def load_translated_topics(translated_file: str) -> dict:
    """Load translated topics from JSON file."""
    with open(translated_file, "r", encoding="utf-8") as f:
        translated_data = json.load(f)
    # Convert to dict by topic number for easy lookup
    translated_dict = {item["number"]: item for item in translated_data}
    return translated_dict


def load_dataset(dataset_file: str) -> list:
    """Load topics from dataset JSON file."""

    with open(dataset_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("topics", [])


async def run_evaluation(
    dataset_file: str, language: str = "en", output_file_name: str = "matching_results_workflow.json"
):
    """Run evaluation on all topics with LLM-as-Judge."""
    print("=" * 80)
    print("Clinical Trial Matching Evaluation")
    print(f"Language: {language.upper()}")
    print("(with LLM-as-Judge automatic scoring)")
    print("=" * 80)

    # Load topics from dataset
    print(f"\nLoading topics from {dataset_file}...")
    topics = load_dataset(dataset_file)
    if not topics:
        print("Error: No topics loaded from dataset")
        return

    topics = topics[:2]

    print(f"Found {len(topics)} topics to evaluate\n")

    # Initialize workflow service
    workflow_service = WorkflowService()

    # Run evaluation for each topic with WorkflowService
    print("Running WorkflowService evaluation...")
    workflow_results = []
    for i, topic in enumerate(topics, 1):
        print(f"[{i}/{len(topics)}] ", end="")
        result = await run_single_topic(topic, workflow_service, language=language)
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
                "language_correction": round(avg_l, 2) if avg_l > 0 else None,
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
        "total_topics": len(topics),
        "results": workflow_results,
        "average_scores": workflow_average_scores,
    }

    with open(workflow_output_file, "w", encoding="utf-8") as f:
        json.dump(workflow_output_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("✓ Evaluation complete!")
    print(f"WorkflowService results saved to: {workflow_output_file}")
    print(f"Total topics: {len(topics)}")
    print("=" * 80 + "\n")

    return str(workflow_output_file)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Clinical Trial Matching Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--lang",
        "-l",
        choices=["en", "vi"],
        help="Language for evaluation: 'en' for English, 'vi' for Vietnamese",
    )

    args = parser.parse_args()

    dataset_file = "benchmark/datasets/01_matching_dataset.json"
    output_file_name = f"matching_results_workflow_{args.lang}_tmp.json"
    # Evaluation mode
    asyncio.run(run_evaluation(dataset_file=dataset_file, language=args.lang, output_file_name=output_file_name))


if __name__ == "__main__":
    main()
