#!/usr/bin/env python3
"""Evaluation script for trial summarization feature using 50 trial IDs."""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from src.api.services.workflow import WorkflowService
from src.config.settings import settings

TRIAL_IDS = [
    "NCT02132585",
    "NCT03341637",
    "NCT00480792",
    "NCT01079793",
    "NCT01670110",
    "NCT00966875",
    "NCT03087305",
    "NCT04888793",
    "NCT01815333",
    "NCT00399217",
    "NCT03966742",
    "NCT00601666",
    "NCT05679622",
    "NCT03701048",
    "NCT01757743",
    "NCT01711658",
    "NCT04912570",
    "NCT04130009",
    "NCT02152072",
    "NCT03835286",
    "NCT05316311",
    "NCT01876264",
    "NCT00679497",
    "NCT00291018",
    "NCT04498546",
    "NCT05611359",
    "NCT02967120",
    "NCT04541875",
    "NCT00250354",
    "NCT05851157",
    "NCT04086849",
    "NCT05735821",
    "NCT01478932",
    "NCT05758727",
    "NCT01061502",
    "NCT02708654",
    "NCT01622413",
    "NCT00473707",
    "NCT01940198",
    "NCT00274963",
    "NCT00005646",
    "NCT00004833",
    "NCT02047695",
    "NCT01301989",
    "NCT03061903",
    "NCT05153993",
    "NCT00195767",
    "NCT00213486",
    "NCT00485602",
    "NCT05146674",
]
TRIAL_IDS = TRIAL_IDS[:2]


def format_response(result: dict) -> str:
    """Extract the response from workflow result."""
    if result.get("chitchat_response"):
        return result.get("chitchat_response", "")
    elif result.get("final_answer"):
        return result.get("final_answer", "")
    else:
        return "No response generated."


def load_llm_judge_prompt() -> str:
    """Load the LLM judge prompt template for summarization."""
    prompt_file = Path("benchmark/llm_judge_summarize_prompt.txt")
    return prompt_file.read_text(encoding="utf-8")


async def llm_evaluate_summarization(trial_id: str, response: str) -> dict:
    """Use LLM to evaluate the summarization quality."""
    # Load and format prompt
    prompt_template = load_llm_judge_prompt()
    prompt = prompt_template.format(trial_id=trial_id, response=response)

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


async def run_single_trial(trial_id: str, workflow_service: WorkflowService) -> dict:
    """Run workflow for a single trial ID and capture results."""
    # Format query to trigger summarization
    query = f"Summarize {trial_id}"

    print(f"Running trial {trial_id}...", end=" ", flush=True)

    start_time = datetime.now()
    result_data = None

    try:
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
        }

        # Always run LLM evaluation
        if result_data:
            llm_scores = await llm_evaluate_summarization(trial_id, response)
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
    """Run evaluation on all trial IDs."""
    print("=" * 80)
    print("Clinical Trial Summarization Evaluation")
    print("(with LLM-as-Judge automatic scoring)")
    print("=" * 80)

    if not TRIAL_IDS:
        print("\n❌ Error: No trial IDs provided!")
        print("Please add 50 trial IDs to the TRIAL_IDS list in the script.")
        return

    print(f"\nFound {len(TRIAL_IDS)} trials to evaluate\n")

    # Initialize workflow service
    workflow_service = WorkflowService()

    # Run evaluation for each trial
    results = []
    for i, trial_id in enumerate(TRIAL_IDS, 1):
        print(f"[{i}/{len(TRIAL_IDS)}] ", end="")
        result = await run_single_trial(trial_id, workflow_service)
        results.append(result)

    # Save results
    output_dir = Path("benchmark/results")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "summarize_results.json"

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "total_trials": len(TRIAL_IDS),
        "results": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("✓ Evaluation complete!")
    print(f"Results saved to: {output_file}")
    print(f"Total trials: {len(TRIAL_IDS)}")
    print("=" * 80 + "\n")

    return str(output_file)


def review_results(results_file: str):
    """Display results with LLM scores."""
    print("=" * 80)
    print("Summarization Results Review")
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
        results_file = Path("benchmark/results/summarize_results.json")
        if not results_file.exists():
            print("No results file found. Run evaluation first.")
            return

        review_results(str(results_file))
    else:
        # Evaluation mode (always uses LLM judge)
        asyncio.run(run_evaluation())


if __name__ == "__main__":
    main()
