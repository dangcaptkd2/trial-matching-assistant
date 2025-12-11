#!/usr/bin/env python3
"""Evaluation script for explain criteria feature using 50 random trial IDs from Elasticsearch."""

import argparse
import asyncio
import json
import random
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

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


class ExplanationEvaluation(BaseModel):
    """Pydantic model for criteria explanation evaluation scores."""

    hallucination: int = Field(
        description="Hallucination score (1-5 scale, where 5 is BEST and 1 is WORST). Score whether all information is accurate and grounded in the provided eligibility criteria."
    )
    hallucination_reasoning: str = Field(description="Brief explanation for the hallucination score")
    accuracy: int = Field(
        description="Accuracy score (1-5 scale, where 5 is BEST and 1 is WORST). Score how accurately the explanation captures all key inclusion and exclusion criteria."
    )
    accuracy_reasoning: str = Field(description="Brief explanation for the accuracy score")
    clarity: int = Field(
        description="Clarity score (1-5 scale, where 5 is BEST and 1 is WORST). Score how clear, well-organized, and patient-friendly the explanation is."
    )
    clarity_reasoning: str = Field(description="Brief explanation for the clarity score")
    language_correction: int = Field(
        description="Language correction score (1-5 scale, where 5 is BEST and 1 is WORST). Score how well the response language matches the user input language."
    )
    language_correction_reasoning: str = Field(
        description="Brief explanation for the language correction score, noting the languages of user input and response"
    )


def load_llm_judge_prompt() -> str:
    """Load the LLM judge prompt template for explain criteria evaluation."""
    prompt_file = Path("benchmark/prompts/04_llm_judge_explain_criteria_prompt.txt")
    return prompt_file.read_text(encoding="utf-8")


async def llm_evaluate_explanation(
    trial_id: str, trial_title: str, eligibility_criteria: str, response: str, user_input: str = ""
) -> dict:
    """Use LLM to evaluate the explanation quality."""
    # Load and format prompt
    prompt_template = load_llm_judge_prompt()
    prompt = prompt_template.format(
        user_input=user_input,
        trial_id=trial_id,
        trial_title=trial_title,
        eligibility_criteria=eligibility_criteria,
        response=response,
    )

    # Initialize the model with structured output
    llm = ChatOpenAI(model=settings.llm_judge_model, temperature=0.0)
    structured_llm = llm.with_structured_output(ExplanationEvaluation)

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
        llm_scores = await llm_evaluate_explanation(
            trial_id,
            trial_data.get("title", "N/A"),
            trial_data.get("eligibility_criteria", ""),
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


async def run_evaluation(language: str = "en"):
    """Run evaluation on random trial IDs from Elasticsearch."""
    print("=" * 80)
    print("Clinical Trial Criteria Explanation Evaluation")
    print(f"Language: {language.upper()}")
    print("(with LLM-as-Judge automatic scoring)")
    print("=" * 80)

    # Initialize services
    es_searcher = ElasticsearchTrialSearcher(index_name=settings.es_index_name)
    workflow_service = WorkflowService()

    # Get random trial IDs
    print("\nFetching random trial IDs from Elasticsearch...")
    trial_ids = await get_random_trial_ids(es_searcher, count=10, seed=42)
    print(f"Found {len(trial_ids)} trials to evaluate\n")

    if not trial_ids:
        print("❌ Error: No trial IDs found!")
        return

    # Run evaluation for each trial
    results = []
    for i, trial_id in enumerate(trial_ids, 1):
        print(f"[{i}/{len(trial_ids)}] ", end="")
        result = await run_single_trial(trial_id, workflow_service, es_searcher, language=language)
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
    output_file = output_dir / f"explain_criteria_results_{language}.json"

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "language": language,
        "total_trials": len(trial_ids),
        "results": results,
        "average_scores": average_scores,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("✓ Evaluation complete!")
    print(f"Results saved to: {output_file}")
    print(f"Total trials: {len(trial_ids)}")
    print("=" * 80 + "\n")

    return str(output_file)


def review_results(results_file: str):
    """Display results with LLM scores."""
    print("=" * 80)
    print("Criteria Explanation Results Review")
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
        print(f"Result {i}/{len(results)} - Trial {result['trial_id']}")
        print("=" * 80)
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
    # Evaluation mode (always uses LLM judge)
    asyncio.run(run_evaluation(language=args.lang))


if __name__ == "__main__":
    main()
