#!/usr/bin/env python3
"""Evaluation script for eligibility checking feature using topics2023.xml and qrels2022.txt."""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

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


class EligibilityEvaluation(BaseModel):
    """Pydantic model for eligibility evaluation scores."""

    hallucination: int = Field(
        description="Hallucination score (1-5 scale, where 5 is BEST and 1 is WORST). Score whether all information is grounded in the provided patient profile and trial eligibility criteria."
    )
    hallucination_reasoning: str = Field(description="Brief explanation for the hallucination score")
    accuracy: int = Field(
        description="Accuracy score (1-5 scale, where 5 is BEST and 1 is WORST). Score whether the eligibility determination is correct and matches ground truth, with accurate reasoning."
    )
    accuracy_reasoning: str = Field(
        description="Brief explanation for the accuracy score, noting if determination matches ground truth"
    )
    clarity: int = Field(
        description="Clarity score (1-5 scale, where 5 is BEST and 1 is WORST). Score how clear, well-organized, and easy to understand the response is."
    )
    clarity_reasoning: str = Field(description="Brief explanation for the clarity score")
    language_correction: int = Field(
        description="Language correction score (1-5 scale, where 5 is BEST and 1 is WORST). Score how well the response language matches the user input language."
    )
    language_correction_reasoning: str = Field(
        description="Brief explanation for the language correction score, noting the languages of user input and response"
    )


def load_llm_judge_prompt() -> str:
    """Load the LLM judge prompt template for eligibility evaluation."""
    prompt_file = Path("benchmark/prompts/03_llm_judge_eligibility_prompt.txt")
    return prompt_file.read_text(encoding="utf-8")


async def llm_evaluate_eligibility(
    patient_profile: str,
    trial_data: dict,
    ground_truth_label: int,
    workflow_answer: str,
    user_input: str = "",
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
        user_input=user_input,
        patient_profile=patient_profile,
        trial_id=trial_id,
        trial_title=trial_title,
        eligibility_criteria=eligibility_criteria,
        ground_truth_label=f"{ground_truth_label} ({label_name})",
        ground_truth_explanation=label_explanation,
        workflow_answer=workflow_answer,
    )

    # Initialize the model with structured output
    llm = ChatOpenAI(model=settings.llm_judge_model, temperature=0.0)
    structured_llm = llm.with_structured_output(EligibilityEvaluation)

    try:
        # Get structured output
        evaluation_result = await structured_llm.ainvoke([HumanMessage(content=prompt)])

        return {
            "hallucination": evaluation_result.hallucination,
            "hallucination_reasoning": evaluation_result.hallucination_reasoning,
            "accuracy": evaluation_result.accuracy,
            "accuracy_reasoning": evaluation_result.accuracy_reasoning,
            "clarity": evaluation_result.clarity,
            "clarity_reasoning": evaluation_result.clarity_reasoning,
            "language_correction": evaluation_result.language_correction,
            "language_correction_reasoning": evaluation_result.language_correction_reasoning,
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
            "language_correction": None,
            "language_correction_reasoning": f"Error: {str(e)}",
        }


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
            "language": language,
        }

        # Always run LLM evaluation
        if result_data and "error" not in trial_data:
            llm_scores = await llm_evaluate_eligibility(
                patient_profile,
                trial_data,
                ground_truth_label,
                response,
                user_input=query,
            )
            evaluation_result["llm_scores"] = llm_scores
            print(
                f"✓ ({elapsed:.1f}s) [LLM: H={llm_scores.get('hallucination', '?')} A={llm_scores.get('accuracy', '?')} C={llm_scores.get('clarity', '?')} L={llm_scores.get('language_correction', '?')}]"
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
            "language": language,
        }


async def run_evaluation(
    topics_file: str = "/Users/quyen.nguyen/Personal/uit/trec_2023/topics2023.xml",
    qrels_file: str = "data/qrels2022.txt",
    language: str = "en",
):
    """Run evaluation on sampled topic-trial pairs."""
    print("=" * 80)
    print("Clinical Trial Eligibility Evaluation")
    print(f"Language: {language.upper()}")
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
            language=language,
        )
        results.append(result)

    # Calculate average scores
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

        average_scores = {
            "overall": {
                "hallucination": round(avg_h, 2),
                "accuracy": round(avg_a, 2),
                "clarity": round(avg_c, 2),
                "language_correction": round(avg_l, 2) if avg_l is not None else None,
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

    # Save results
    output_dir = Path("benchmark/results")
    output_dir.mkdir(exist_ok=True)

    # Include language in filename to avoid overwriting
    output_file = output_dir / f"eligibility_results_{language}.json"

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "topics_file": topics_file,
        "qrels_file": qrels_file,
        "language": language,
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
            print(
                f"  Language Correction: {llm_scores.get('language_correction', '?')}/5 - {llm_scores.get('language_correction_reasoning', '')[:80]}..."
            )
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
        lang_corr_scored = [r for r in llm_scored if r["llm_scores"].get("language_correction") is not None]
        avg_l = (
            sum(r["llm_scores"]["language_correction"] for r in lang_corr_scored) / len(lang_corr_scored)
            if lang_corr_scored
            else None
        )
        print(f"  Average Hallucination: {avg_h:.2f}")
        print(f"  Average Accuracy: {avg_a:.2f}")
        print(f"  Average Clarity: {avg_c:.2f}")
        if avg_l is not None:
            print(f"  Average Language Correction: {avg_l:.2f}")

        # Group by ground truth label
        print("\n📊 Scores by Ground Truth Label:")
        for label in [0, 1, 2]:
            label_name, _ = get_ground_truth_label_info(label)
            label_results = [r for r in llm_scored if r["ground_truth_label"] == label]
            if label_results:
                avg_h_label = sum(r["llm_scores"]["hallucination"] for r in label_results) / len(label_results)
                avg_a_label = sum(r["llm_scores"]["accuracy"] for r in label_results) / len(label_results)
                avg_c_label = sum(r["llm_scores"]["clarity"] for r in label_results) / len(label_results)
                lang_corr_label = [r for r in label_results if r["llm_scores"].get("language_correction") is not None]
                avg_l_label = (
                    sum(r["llm_scores"]["language_correction"] for r in lang_corr_label) / len(lang_corr_label)
                    if lang_corr_label
                    else None
                )
                print(f"  Label {label} ({label_name}) - {len(label_results)} samples:")
                print(f"    Avg H: {avg_h_label:.2f}, Avg A: {avg_a_label:.2f}, Avg C: {avg_c_label:.2f}", end="")
                if avg_l_label is not None:
                    print(f", Avg L: {avg_l_label:.2f}")
                else:
                    print()

        print("=" * 80)


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

    # Evaluation mode (always uses LLM judge)
    asyncio.run(run_evaluation(language=args.lang))


if __name__ == "__main__":
    main()
