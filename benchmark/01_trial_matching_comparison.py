"""Evaluation script for comparing ChatGPT vs Workflow for clinical trial matching."""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from benchmark.competitor_agent import gpt_invoke, load_dataset
from benchmark.utilis import Task, extract_trial_ids_from_text, llm_evaluate_response
from src.api.services.workflow import WorkflowService


def format_response(result: dict) -> str:
    """Extract the response from workflow result."""
    if result.get("chitchat_response"):
        return result.get("chitchat_response", "")
    elif result.get("final_answer"):
        return result.get("final_answer", "")
    else:
        return "No response generated."


async def run_single_topic_comparison(topic: dict, workflow_service: WorkflowService, language: str = "en") -> dict:
    """Run both ChatGPT and workflow for a single topic and capture results."""
    topic_number = topic["number"]

    # Get patient profile based on language
    if language == "vi":
        patient_profile = topic.get("vi_content", topic.get("content", ""))
        query = "tìm thử nghiệm lâm sàng cho bệnh nhân này: " + patient_profile
    else:
        patient_profile = topic.get("content", "")
        query = "find trials for this patient: " + patient_profile

    print(f"Running topic {topic_number} ({language})...", end=" ", flush=True)

    # Run ChatGPT
    chatgpt_start = datetime.now()
    chatgpt_response_text = await gpt_invoke(query)
    chatgpt_elapsed = (datetime.now() - chatgpt_start).total_seconds()

    # Extract trial IDs from ChatGPT response
    chatgpt_trial_ids = extract_trial_ids_from_text(chatgpt_response_text)
    chatgpt_trial_count = len(chatgpt_trial_ids)

    # Run Workflow
    workflow_start = datetime.now()
    workflow_result_data = None
    async for event in workflow_service.invoke_workflow(
        user_input=query,
        thread_id=f"comparison-topic-{topic_number}-{language}",
        top_k=10,
        stream=False,
    ):
        if event["type"] == "result":
            workflow_result_data = event.get("data", {})

    workflow_elapsed = (datetime.now() - workflow_start).total_seconds()
    workflow_response = format_response(workflow_result_data) if workflow_result_data else "Error"
    workflow_trial_ids = extract_trial_ids_from_text(workflow_response)
    workflow_trial_count = len(workflow_trial_ids)

    # Run LLM evaluation for ChatGPT
    chatgpt_llm_scores = {}
    if chatgpt_response_text:
        chatgpt_llm_scores = await llm_evaluate_response(
            task=Task.MATCHING_COMPARISON,
            response=chatgpt_response_text,
            user_input=query,
        )

    # Run LLM evaluation for Workflow
    workflow_llm_scores = {}
    if workflow_result_data:
        workflow_llm_scores = await llm_evaluate_response(
            task=Task.MATCHING_COMPARISON,
            response=workflow_response,
            user_input=query,
        )

    evaluation_result = {
        "topic_number": topic_number,
        "chatgpt": {
            "response": chatgpt_response_text,
            "trial_ids_extracted": chatgpt_trial_ids,
            "trial_count": chatgpt_trial_count,
            "execution_time": round(chatgpt_elapsed, 2),
            "llm_scores": chatgpt_llm_scores,
        },
        "workflow": {
            "response": workflow_response,
            "trial_ids": workflow_trial_ids,
            "trial_count": workflow_trial_count,
            "execution_time": round(workflow_elapsed, 2),
            "llm_scores": workflow_llm_scores,
        },
        "language": language,
    }

    # Print summary
    print(
        f"✓ [ChatGPT: {chatgpt_trial_count} trials, {chatgpt_elapsed:.1f}s] "
        f"[Workflow: {workflow_trial_count} trials, {workflow_elapsed:.1f}s]"
    )

    return evaluation_result


async def run_evaluation(
    dataset_file: str, language: str = "en", output_file_name: str = "matching_comparison_results.json"
):
    """Run comparison evaluation on all topics."""
    print("=" * 80)
    print("Clinical Trial Matching Comparison Evaluation")
    print("ChatGPT vs Workflow")
    print(f"Language: {language.upper()}")
    print("(with LLM-as-Judge automatic scoring)")
    print("=" * 80)

    # Load topics from dataset
    print(f"\nLoading topics from {dataset_file}...")
    topics = load_dataset(dataset_file)
    if not topics:
        print("Error: No topics loaded from dataset")
        return

    # Limit topics for testing (remove this line for full evaluation)
    topics = topics[:10]

    print(f"Found {len(topics)} topics to evaluate\n")

    # Initialize workflow service
    workflow_service = WorkflowService()

    # Run evaluation for each topic
    print("Running comparison evaluation...")
    results = []
    for i, topic in enumerate(topics, 1):
        print(f"[{i}/{len(topics)}] ", end="")
        result = await run_single_topic_comparison(topic, workflow_service, language=language)
        results.append(result)

    # Calculate average scores
    chatgpt_scored = [r for r in results if r.get("chatgpt", {}).get("llm_scores", {}).get("depth") is not None]
    workflow_scored = [r for r in results if r.get("workflow", {}).get("llm_scores", {}).get("depth") is not None]

    average_scores = {}

    if chatgpt_scored:
        chatgpt_avg = {}
        metrics = ["trial_id_count", "depth", "relevance", "clarity", "completeness"]
        for metric in metrics:
            scores = [
                r["chatgpt"]["llm_scores"].get(metric)
                for r in chatgpt_scored
                if r["chatgpt"]["llm_scores"].get(metric) is not None
            ]
            if scores:
                chatgpt_avg[metric] = round(sum(scores) / len(scores), 2)
        chatgpt_avg["trial_count_avg"] = round(sum(r["chatgpt"]["trial_count"] for r in results) / len(results), 2)
        chatgpt_avg["execution_time_avg"] = round(
            sum(r["chatgpt"]["execution_time"] for r in results) / len(results), 2
        )
        chatgpt_avg["total_scored"] = len(chatgpt_scored)
        average_scores["chatgpt"] = chatgpt_avg

    if workflow_scored:
        workflow_avg = {}
        metrics = ["trial_id_count", "depth", "relevance", "clarity", "completeness"]
        for metric in metrics:
            scores = [
                r["workflow"]["llm_scores"].get(metric)
                for r in workflow_scored
                if r["workflow"]["llm_scores"].get(metric) is not None
            ]
            if scores:
                workflow_avg[metric] = round(sum(scores) / len(scores), 2)
        workflow_avg["trial_count_avg"] = round(sum(r["workflow"]["trial_count"] for r in results) / len(results), 2)
        workflow_avg["execution_time_avg"] = round(
            sum(r["workflow"]["execution_time"] for r in results) / len(results), 2
        )
        workflow_avg["total_scored"] = len(workflow_scored)
        average_scores["workflow"] = workflow_avg

    # Save results
    output_dir = Path("benchmark/results")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / output_file_name
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "dataset_file": dataset_file,
        "evaluation_type": "comparison",
        "language": language,
        "total_topics": len(topics),
        "results": results,
        "average_scores": average_scores,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("✓ Evaluation complete!")
    print(f"Results saved to: {output_file}")
    print(f"Total topics: {len(topics)}")
    if average_scores:
        print("\nAverage Scores:")
        if "chatgpt" in average_scores:
            print(f"  ChatGPT: {average_scores['chatgpt']}")
        if "workflow" in average_scores:
            print(f"  Workflow: {average_scores['workflow']}")
    print("=" * 80 + "\n")

    return str(output_file)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Clinical Trial Matching Comparison Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--lang",
        "-l",
        choices=["en", "vi"],
        default="vi",
        help="Language for evaluation: 'en' for English, 'vi' for Vietnamese",
    )

    args = parser.parse_args()

    dataset_file = "benchmark/datasets/01_matching_dataset.json"
    output_file_name = f"01_matching_comparison_results_{args.lang}.json"
    asyncio.run(run_evaluation(dataset_file=dataset_file, language=args.lang, output_file_name=output_file_name))


if __name__ == "__main__":
    main()
