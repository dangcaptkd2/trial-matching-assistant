#!/usr/bin/env python3
"""Evaluation script for explain criteria feature using 50 random trial IDs from Elasticsearch."""

import asyncio
import json
import random
import sys
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

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
    try:
        doc = es_searcher.es.client.get(index=es_searcher.index_name, id=trial_id)
        source = doc.get("_source", {})
        return {
            "nct_id": trial_id,
            "title": source.get("brief_title") or source.get("official_title", "N/A"),
            "eligibility_criteria": source.get("eligibility_criteria", ""),
        }
    except Exception as e:
        return {
            "nct_id": trial_id,
            "error": f"Trial {trial_id} not found: {str(e)}",
        }


def load_llm_judge_prompt() -> str:
    """Load the LLM judge prompt template for explain criteria evaluation."""
    prompt_file = Path("benchmark/prompts/llm_judge_explain_criteria_prompt.txt")
    return prompt_file.read_text(encoding="utf-8")


async def llm_evaluate_explanation(trial_id: str, trial_title: str, eligibility_criteria: str, response: str) -> dict:
    """Use LLM to evaluate the explanation quality."""
    # Load and format prompt
    prompt_template = load_llm_judge_prompt()
    prompt = prompt_template.format(
        trial_id=trial_id,
        trial_title=trial_title,
        eligibility_criteria=eligibility_criteria,
        response=response,
    )

    # Call LLM
    llm = ChatOpenAI(model=settings.llm_model, temperature=0.0)

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


async def run_single_trial(
    trial_id: str, workflow_service: WorkflowService, es_searcher: ElasticsearchTrialSearcher
) -> dict:
    """Run workflow for a single trial ID and capture results."""
    # Format query to trigger explain criteria
    query = f"Explain the eligibility criteria for trial {trial_id}"

    print(f"Running trial {trial_id}...", end=" ", flush=True)

    start_time = datetime.now()
    result_data = None

    try:
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
        }

        # Always run LLM evaluation
        if result_data and "error" not in trial_data:
            llm_scores = await llm_evaluate_explanation(
                trial_id,
                trial_data.get("title", "N/A"),
                trial_data.get("eligibility_criteria", ""),
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
            "trial_id": trial_id,
            "query": query,
            "response": f"ERROR: {str(e)}",
            "execution_time": elapsed,
            "error": str(e),
        }


async def run_evaluation():
    """Run evaluation on random trial IDs from Elasticsearch."""
    print("=" * 80)
    print("Clinical Trial Criteria Explanation Evaluation")
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
        result = await run_single_trial(trial_id, workflow_service, es_searcher)
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

    # Save results
    output_dir = Path("benchmark/results")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "explain_criteria_results.json"

    output_data = {
        "timestamp": datetime.now().isoformat(),
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
        print("=" * 80)


def main():
    """Main entry point."""

    if len(sys.argv) > 1 and sys.argv[1] == "review":
        # Review mode
        results_file = Path("benchmark/results/explain_criteria_results.json")
        if not results_file.exists():
            print("No results file found. Run evaluation first.")
            return

        review_results(str(results_file))
    else:
        # Evaluation mode (always uses LLM judge)
        asyncio.run(run_evaluation())


if __name__ == "__main__":
    main()
