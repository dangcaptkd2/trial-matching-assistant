#!/usr/bin/env python3
"""Evaluation script for compare trials feature using qrels2022.txt."""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from benchmark.xml_data_utils import parse_qrels_file, sample_trials_for_comparison
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


async def fetch_trial_data(trial_id: str, es_searcher: ElasticsearchTrialSearcher) -> dict:
    """Fetch trial data from Elasticsearch."""
    try:
        doc = es_searcher.es.client.get(index=es_searcher.index_name, id=trial_id)
        source = doc.get("_source", {})
        return {
            "nct_id": trial_id,
            "title": source.get("brief_title") or source.get("official_title", "N/A"),
            "phase": source.get("phase", "N/A"),
            "status": source.get("overall_status", "N/A"),
            "eligibility_criteria": source.get("eligibility_criteria", ""),
            "brief_summary": source.get("brief_summary", ""),
            "detailed_description": source.get("detailed_description", ""),
            "locations": source.get("locations", "N/A"),
            "start_date": source.get("start_date", "N/A"),
            "completion_date": source.get("completion_date", "N/A"),
            "primary_outcome": source.get("primary_outcome_measure", "N/A"),
        }
    except Exception as e:
        return {
            "nct_id": trial_id,
            "error": f"Trial {trial_id} not found: {str(e)}",
        }


def format_trial_information_for_judge(trial_data_list: list) -> str:
    """Format trial data for LLM judge prompt."""
    trials_text = ""
    for trial in trial_data_list:
        if "error" in trial:
            trials_text += f"\n- {trial['nct_id']}: {trial['error']}\n"
        else:
            trials_text += f"\nTrial ID: {trial.get('nct_id', 'N/A')}\n"
            trials_text += f"Title: {trial.get('title', 'N/A')}\n"
            trials_text += f"Phase: {trial.get('phase', 'N/A')}\n"
            trials_text += f"Status: {trial.get('status', 'N/A')}\n"
            if trial.get("brief_summary"):
                trials_text += f"Summary: {trial.get('brief_summary')}\n"
            if trial.get("eligibility_criteria"):
                trials_text += f"Eligibility: {trial.get('eligibility_criteria')}\n"
            if trial.get("locations") and trial.get("locations") != "N/A":
                trials_text += f"Locations: {trial.get('locations')}\n"
            if trial.get("primary_outcome") and trial.get("primary_outcome") != "N/A":
                trials_text += f"Primary Outcome: {trial.get('primary_outcome')}\n"
    return trials_text


class ComparisonEvaluation(BaseModel):
    """Pydantic model for trial comparison evaluation scores."""

    hallucination: int = Field(
        description="Hallucination score (1-5 scale, where 5 is BEST and 1 is WORST). Score whether all information is accurate and grounded in the provided trial information."
    )
    hallucination_reasoning: str = Field(description="Brief explanation for the hallucination score")
    accuracy: int = Field(
        description="Accuracy score (1-5 scale, where 5 is BEST and 1 is WORST). Score how accurately the comparison captures all key differences and similarities between trials."
    )
    accuracy_reasoning: str = Field(description="Brief explanation for the accuracy score")
    clarity: int = Field(
        description="Clarity score (1-5 scale, where 5 is BEST and 1 is WORST). Score how clear, well-organized, and patient-friendly the comparison is."
    )
    clarity_reasoning: str = Field(description="Brief explanation for the clarity score")
    language_correction: int = Field(
        description="Language correction score (1-5 scale, where 5 is BEST and 1 is WORST). Score how well the response language matches the user input language."
    )
    language_correction_reasoning: str = Field(
        description="Brief explanation for the language correction score, noting the languages of user input and response"
    )


def load_llm_judge_prompt() -> str:
    """Load the LLM judge prompt template for compare trials evaluation."""
    prompt_file = Path("benchmark/prompts/06_llm_judge_compare_trials_prompt.txt")
    return prompt_file.read_text(encoding="utf-8")


async def llm_evaluate_comparison(trial_information: str, response: str, user_input: str = "") -> dict:
    """Use LLM to evaluate the comparison quality."""
    # Load and format prompt
    prompt_template = load_llm_judge_prompt()
    prompt = prompt_template.format(user_input=user_input, trial_information=trial_information, response=response)

    # Initialize the model with structured output
    llm = ChatOpenAI(model=settings.llm_judge_model, temperature=0.0)
    structured_llm = llm.with_structured_output(ComparisonEvaluation)

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


async def run_single_comparison(
    topic_number: str,
    trial_ids: list[str],
    workflow_service: WorkflowService,
    es_searcher: ElasticsearchTrialSearcher,
    language: str = "en",
) -> dict:
    """Run workflow for a single comparison and capture results."""
    # Format query to trigger comparison based on language
    trial_ids_str = ", ".join(trial_ids)
    if language == "vi":
        query = f"So sánh các thử nghiệm này: {trial_ids_str}"
    else:
        query = f"Compare these trials: {trial_ids_str}"

    print(f"Running topic {topic_number} with {len(trial_ids)} trials...", end=" ", flush=True)

    start_time = datetime.now()
    result_data = None

    try:
        # Fetch trial data for LLM judge
        trial_data_list = []
        for trial_id in trial_ids:
            trial_data = await fetch_trial_data(trial_id, es_searcher)
            trial_data_list.append(trial_data)

        # Check if we have valid trials
        valid_trials = [t for t in trial_data_list if "error" not in t]
        if len(valid_trials) < 2:
            print(f"⚠️  Only {len(valid_trials)} valid trials found, skipping")
            return {
                "topic_number": topic_number,
                "trial_ids": trial_ids,
                "query": query,
                "response": "ERROR: Not enough valid trials",
                "execution_time": (datetime.now() - start_time).total_seconds(),
                "error": "Not enough valid trials",
                "language": language,
            }

        # Run workflow
        async for event in workflow_service.invoke_workflow(
            user_input=query,
            thread_id=f"eval-compare-{topic_number}",
            top_k=10,
            stream=False,
        ):
            if event["type"] == "result":
                result_data = event.get("data", {})

        elapsed = (datetime.now() - start_time).total_seconds()

        response = format_response(result_data) if result_data else "Error"

        evaluation_result = {
            "topic_number": topic_number,
            "trial_ids": trial_ids,
            "query": query,
            "response": response,
            "trial_data": trial_data_list,
            "execution_time": elapsed,
            "language": language,
        }

        # Always run LLM evaluation
        if result_data and len(valid_trials) >= 2:
            trial_information = format_trial_information_for_judge(trial_data_list)
            llm_scores = await llm_evaluate_comparison(trial_information, response, user_input=query)
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
            "trial_ids": trial_ids,
            "query": query,
            "response": f"ERROR: {str(e)}",
            "execution_time": elapsed,
            "error": str(e),
            "language": language,
        }


async def run_evaluation(qrels_file: str = "data/qrels2022.txt", num_topics: int = 10, language: str = "en"):
    """Run evaluation on sampled topic-trial groups."""
    print("=" * 80)
    print("Clinical Trial Comparison Evaluation")
    print(f"Language: {language.upper()}")
    print("(with LLM-as-Judge automatic scoring)")
    print("=" * 80)

    # Load qrels
    print(f"\nLoading qrels from {qrels_file}...")
    qrels = parse_qrels_file(qrels_file)
    print(f"Found {len(qrels)} qrels entries")

    # Sample topics with 2-3 eligible trials each
    print(f"\nSampling {num_topics} topics with 2-3 eligible trials each...")
    samples = sample_trials_for_comparison(qrels, num_topics=num_topics, min_trials=2, max_trials=3, seed=42)
    print(f"Sampled {len(samples)} topic groups:")
    trial_counts = {}
    for _, trials in samples:
        count = len(trials)
        trial_counts[count] = trial_counts.get(count, 0) + 1
    for count, num in sorted(trial_counts.items()):
        print(f"  {count} trials: {num} groups")

    # Initialize services
    workflow_service = WorkflowService()
    es_searcher = ElasticsearchTrialSearcher(index_name=settings.es_index_name)

    # Run evaluation for each sample
    results = []
    for i, (topic_number, trial_ids) in enumerate(samples, 1):
        print(f"\n[{i}/{len(samples)}] ", end="")
        result = await run_single_comparison(topic_number, trial_ids, workflow_service, es_searcher, language=language)
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

    # Save results
    output_dir = Path("benchmark/results")
    output_dir.mkdir(exist_ok=True)

    # Include language in filename to avoid overwriting
    output_file = output_dir / f"compare_trials_results_{language}.json"

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "qrels_file": qrels_file,
        "language": language,
        "num_topics": num_topics,
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
    print("Trial Comparison Results Review")
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
        print(f"Result {i}/{len(results)} - Topic {result['topic_number']}")
        print("=" * 80)
        print(f"\n📋 Trial IDs: {', '.join(result['trial_ids'])}")
        print(f"\n📋 Query: {result['query']}")
        print(f"\n💬 Response:\n{result['response'][:800]}...")

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
        print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Clinical Trial Comparison Evaluation Script",
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
