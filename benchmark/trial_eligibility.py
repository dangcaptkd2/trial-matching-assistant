#!/usr/bin/env python3
"""Evaluation script for eligibility checking feature using topics2023.xml and qrels2022.txt."""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from benchmark.xml_data_utils import (
    parse_qrels_file,
    read_topics2023_xml_file,
    sample_qrels_by_label,
)
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
    try:
        doc = es_searcher.es.client.get(index=es_searcher.index_name, id=trial_id)
        source = doc.get("_source", {})
        return {
            "nct_id": trial_id,
            "title": source.get("brief_title") or source.get("official_title", "N/A"),
            "eligibility_criteria": source.get("eligibility_criteria", ""),
            "brief_summary": source.get("brief_summary", ""),
            "detailed_description": source.get("detailed_description", ""),
        }
    except Exception as e:
        return {
            "nct_id": trial_id,
            "error": f"Trial {trial_id} not found: {str(e)}",
        }


def load_llm_judge_prompt() -> str:
    """Load the LLM judge prompt template for eligibility evaluation."""
    prompt_file = Path("benchmark/prompts/llm_judge_eligibility_prompt.txt")
    return prompt_file.read_text(encoding="utf-8")


async def llm_evaluate_eligibility(
    patient_profile: str,
    trial_data: dict,
    ground_truth_label: int,
    workflow_answer: str,
) -> dict:
    """Use LLM to evaluate the eligibility determination quality."""
    # Get ground truth info
    label_name, label_explanation = get_ground_truth_label_info(ground_truth_label)

    # Format trial information
    trial_id = trial_data.get("nct_id", "N/A")
    trial_title = trial_data.get("title", "N/A")
    eligibility_criteria = trial_data.get("eligibility_criteria", "Not available")

    # Load and format prompt
    prompt_template = load_llm_judge_prompt()
    prompt = prompt_template.format(
        patient_profile=patient_profile,
        trial_id=trial_id,
        trial_title=trial_title,
        eligibility_criteria=eligibility_criteria,
        ground_truth_label=f"{ground_truth_label} ({label_name})",
        ground_truth_explanation=label_explanation,
        workflow_answer=workflow_answer,
    )

    # Call LLM
    llm = ChatOpenAI(model=settings.llm_judge_model, temperature=0.0)

    try:
        response_msg = await llm.ainvoke([HumanMessage(content=prompt)])
        content = response_msg.content.strip()

        # Extract JSON from response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        result = json.loads(content)

        return {
            "hallucination": result.get("hallucination", None),
            "hallucination_reasoning": result.get("hallucination_reasoning", ""),
            "accuracy": result.get("accuracy", None),
            "accuracy_reasoning": result.get("accuracy_reasoning", ""),
            "clarity": result.get("clarity", None),
            "clarity_reasoning": result.get("clarity_reasoning", ""),
        }

    except Exception as e:
        print(f"  LLM evaluation error: {str(e)}")
        return {
            "hallucination": None,
            "hallucination_reasoning": f"Error: {str(e)}",
            "accuracy": None,
            "accuracy_reasoning": f"Error: {str(e)}",
            "clarity": None,
            "clarity_reasoning": f"Error: {str(e)}",
        }


async def run_single_eligibility_check(
    topic_number: str,
    trial_id: str,
    ground_truth_label: int,
    patient_profile: str,
    workflow_service: WorkflowService,
    es_searcher: ElasticsearchTrialSearcher,
) -> dict:
    """Run workflow for a single topic-trial pair and capture results."""
    # Format query to trigger eligibility check
    query = f"Is this patient eligible for {trial_id}? {patient_profile}"

    print(f"Running topic {topic_number} + trial {trial_id} (label={ground_truth_label})...", end=" ", flush=True)

    start_time = datetime.now()
    result_data = None

    try:
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
        }

        # Always run LLM evaluation
        if result_data and "error" not in trial_data:
            llm_scores = await llm_evaluate_eligibility(
                patient_profile,
                trial_data,
                ground_truth_label,
                response,
            )
            evaluation_result["llm_scores"] = llm_scores
            print(
                f"✓ ({elapsed:.1f}s) [LLM: H={llm_scores.get('hallucination', '?')} A={llm_scores.get('accuracy', '?')} C={llm_scores.get('clarity', '?')}]"
            )
        else:
            print(f"✓ ({elapsed:.1f}s)")

        return evaluation_result

    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"✗ Error: {str(e)}")
        return {
            "topic_number": topic_number,
            "trial_id": trial_id,
            "ground_truth_label": ground_truth_label,
            "patient_profile": patient_profile,
            "query": query,
            "response": f"ERROR: {str(e)}",
            "execution_time": elapsed,
            "error": str(e),
        }


async def run_evaluation(
    topics_file: str = "/Users/quyen.nguyen/Personal/uit/trec_2023/topics2023.xml",
    qrels_file: str = "data/qrels2022.txt",
):
    """Run evaluation on sampled topic-trial pairs."""
    print("=" * 80)
    print("Clinical Trial Eligibility Evaluation")
    print("(with LLM-as-Judge automatic scoring)")
    print("=" * 80)

    # Load topics
    print(f"\nLoading topics from {topics_file}...")
    topics_data = read_topics2023_xml_file(topics_file)
    topics_dict = {topic["number"]: topic for topic in topics_data.get("topics", [])}
    print(f"Found {len(topics_dict)} topics")

    # Load qrels
    print(f"\nLoading qrels from {qrels_file}...")
    qrels = parse_qrels_file(qrels_file)
    print(f"Found {len(qrels)} qrels entries")

    # Sample qrels by label
    label_counts = {0: 17, 1: 17, 2: 16}
    samples = sample_qrels_by_label(qrels, label_counts, seed=42)
    samples = samples[:10]
    print(f"\nSampled {len(samples)} topic-trial pairs:")
    for label, _ in label_counts.items():
        label_name, _ = get_ground_truth_label_info(label)
        actual_count = sum(1 for _, _, l in samples if l == label)
        print(f"  Label {label} ({label_name}): {actual_count} samples")

    # Initialize services
    workflow_service = WorkflowService()
    es_searcher = ElasticsearchTrialSearcher(index_name=settings.es_index_name)

    # Run evaluation for each sample
    results = []
    for i, (topic_number, trial_id, label) in enumerate(samples, 1):
        # Get patient profile from topic
        topic = topics_dict.get(topic_number)
        if not topic:
            print(f"\n[{i}/{len(samples)}] ⚠️  Topic {topic_number} not found in topics file, skipping")
            continue

        patient_profile = topic.get("content", "")
        if not patient_profile:
            print(f"\n[{i}/{len(samples)}] ⚠️  Topic {topic_number} has no content, skipping")
            continue

        print(f"\n[{i}/{len(samples)}] ", end="")
        result = await run_single_eligibility_check(
            topic_number,
            trial_id,
            label,
            patient_profile,
            workflow_service,
            es_searcher,
        )
        results.append(result)

    # Calculate average scores
    llm_scored = [r for r in results if r.get("llm_scores", {}).get("hallucination") is not None]

    average_scores = {}
    if llm_scored:
        avg_h = sum(r["llm_scores"]["hallucination"] for r in llm_scored) / len(llm_scored)
        avg_a = sum(r["llm_scores"]["accuracy"] for r in llm_scored) / len(llm_scored)
        avg_c = sum(r["llm_scores"]["clarity"] for r in llm_scored) / len(llm_scored)

        average_scores = {
            "overall": {
                "hallucination": round(avg_h, 2),
                "accuracy": round(avg_a, 2),
                "clarity": round(avg_c, 2),
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

                by_label[str(label)] = {
                    "label_name": label_name,
                    "hallucination": round(avg_h_label, 2),
                    "accuracy": round(avg_a_label, 2),
                    "clarity": round(avg_c_label, 2),
                    "count": len(label_results),
                }

        if by_label:
            average_scores["by_label"] = by_label

    # Save results
    output_dir = Path("benchmark/results")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "eligibility_results.json"

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "topics_file": topics_file,
        "qrels_file": qrels_file,
        "label_counts": label_counts,
        "total_samples": len(samples),
        "results": results,
        "average_scores": average_scores,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("✓ Evaluation complete!")
    print(f"Results saved to: {output_file}")
    print(f"Total samples: {len(samples)}")
    print("=" * 80 + "\n")

    return str(output_file)


def review_results(results_file: str):
    """Display results with LLM scores."""
    print("=" * 80)
    print("Eligibility Evaluation Results Review")
    print("=" * 80)

    # Load results
    with open(results_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    llm_scored = [r for r in results if r.get("llm_scores", {}).get("hallucination") is not None]

    print(f"\nTotal results: {len(results)}")
    print(f"LLM scored: {len(llm_scored)}\n")

    # Display each result
    for i, result in enumerate(results, 1):
        print("=" * 80)
        print(f"Result {i}/{len(results)} - Topic {result['topic_number']} + Trial {result['trial_id']}")
        print("=" * 80)

        label_name, label_explanation = get_ground_truth_label_info(result["ground_truth_label"])
        print(f"\n📋 Ground Truth: Label {result['ground_truth_label']} ({label_name})")
        print(f"   {label_explanation}")

        print(f"\n👤 Patient Profile:\n{result['patient_profile'][:300]}...")
        print(f"\n💬 Workflow Response:\n{result['response'][:500]}...")

        # Show LLM scores
        if result.get("llm_scores"):
            llm_scores = result["llm_scores"]
            print("\n🤖 LLM Judge Scores:")
            print(
                f"  Hallucination: {llm_scores.get('hallucination', '?')}/5 - {llm_scores.get('hallucination_reasoning', '')[:80]}..."
            )
            print(
                f"  Accuracy: {llm_scores.get('accuracy', '?')}/5 - {llm_scores.get('accuracy_reasoning', '')[:80]}..."
            )
            print(f"  Clarity: {llm_scores.get('clarity', '?')}/5 - {llm_scores.get('clarity_reasoning', '')[:80]}...")
        else:
            print("\n⚠️  No LLM scores available")

        print("\n" + "-" * 80)

    # Print summary
    if llm_scored:
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"\n🤖 LLM Scores ({len(llm_scored)}/{len(results)} scored):")
        avg_h = sum(r["llm_scores"]["hallucination"] for r in llm_scored) / len(llm_scored)
        avg_a = sum(r["llm_scores"]["accuracy"] for r in llm_scored) / len(llm_scored)
        avg_c = sum(r["llm_scores"]["clarity"] for r in llm_scored) / len(llm_scored)
        print(f"  Average Hallucination: {avg_h:.2f}")
        print(f"  Average Accuracy: {avg_a:.2f}")
        print(f"  Average Clarity: {avg_c:.2f}")

        # Group by ground truth label
        print("\n📊 Scores by Ground Truth Label:")
        for label in [0, 1, 2]:
            label_name, _ = get_ground_truth_label_info(label)
            label_results = [r for r in llm_scored if r["ground_truth_label"] == label]
            if label_results:
                avg_h_label = sum(r["llm_scores"]["hallucination"] for r in label_results) / len(label_results)
                avg_a_label = sum(r["llm_scores"]["accuracy"] for r in label_results) / len(label_results)
                avg_c_label = sum(r["llm_scores"]["clarity"] for r in label_results) / len(label_results)
                print(f"  Label {label} ({label_name}) - {len(label_results)} samples:")
                print(f"    Avg H: {avg_h_label:.2f}, Avg A: {avg_a_label:.2f}, Avg C: {avg_c_label:.2f}")

        print("=" * 80)


def main():
    """Main entry point."""

    if len(sys.argv) > 1 and sys.argv[1] == "review":
        # Review mode
        results_file = Path("benchmark/results/eligibility_results.json")
        if not results_file.exists():
            print("No results file found. Run evaluation first.")
            return

        review_results(str(results_file))
    else:
        # Evaluation mode (always uses LLM judge)
        asyncio.run(run_evaluation())


if __name__ == "__main__":
    main()
