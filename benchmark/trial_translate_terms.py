#!/usr/bin/env python3
"""Evaluation script for translate terms feature using a list of user queries."""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from src.api.services.workflow import WorkflowService
from src.config.settings import settings

# List of user queries to evaluate (edit this list manually)
# List of user queries to evaluate (edit this list manually)
USER_QUERIES = [
    "What does 'progression-free survival' mean?",
    "Explain EGFR mutation",
    "What do these terms mean: adjuvant therapy and metastasis?",
    "What is chemotherapy?",
    "Explain what 'overall survival' means",
    "What does 'biopsy' mean in cancer diagnosis?",
    "Can you explain what 'tumor marker' means?",
    "What is the meaning of 'immunotherapy'?",
    "What does 'staging' mean in cancer?",
    "Explain what 'remission' means in medical terms",
]


def format_response(result: dict) -> str:
    """Extract the response from workflow result."""
    if result.get("chitchat_response"):
        return result.get("chitchat_response", "")
    elif result.get("final_answer"):
        return result.get("final_answer", "")
    else:
        return "No response generated."


def load_llm_judge_prompt() -> str:
    """Load the LLM judge prompt template for translate terms evaluation."""
    prompt_file = Path("benchmark/prompts/llm_judge_translate_terms_prompt.txt")
    return prompt_file.read_text(encoding="utf-8")


async def llm_evaluate_translation(user_query: str, response: str) -> dict:
    """Use LLM to evaluate the translation quality."""
    # Load and format prompt
    prompt_template = load_llm_judge_prompt()
    prompt = prompt_template.format(user_query=user_query, response=response)

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
            "accuracy": result.get("accuracy", None),
            "accuracy_reasoning": result.get("accuracy_reasoning", ""),
            "clarity": result.get("clarity", None),
            "clarity_reasoning": result.get("clarity_reasoning", ""),
        }

    except Exception as e:
        print(f"  LLM evaluation error: {str(e)}")
        return {
            "accuracy": None,
            "accuracy_reasoning": f"Error: {str(e)}",
            "clarity": None,
            "clarity_reasoning": f"Error: {str(e)}",
        }


async def run_single_query(query: str, workflow_service: WorkflowService) -> dict:
    """Run workflow for a single query and capture results."""
    print(f"Running query: {query[:60]}...", end=" ", flush=True)

    start_time = datetime.now()
    result_data = None

    try:
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
        }

        # Always run LLM evaluation
        if result_data:
            llm_scores = await llm_evaluate_translation(query, response)
            evaluation_result["llm_scores"] = llm_scores
            print(f"✓ ({elapsed:.1f}s) [LLM: A={llm_scores.get('accuracy', '?')} C={llm_scores.get('clarity', '?')}]")
        else:
            print(f"✓ ({elapsed:.1f}s)")

        return evaluation_result

    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"✗ Error: {str(e)}")
        return {
            "query": query,
            "response": f"ERROR: {str(e)}",
            "execution_time": elapsed,
            "error": str(e),
        }


async def run_evaluation():
    """Run evaluation on all user queries."""
    print("=" * 80)
    print("Medical Terms Translation Evaluation")
    print("(with LLM-as-Judge automatic scoring)")
    print("=" * 80)

    if not USER_QUERIES:
        print("\n❌ Error: No user queries provided!")
        print("Please add queries to the USER_QUERIES list in the script.")
        return

    print(f"\nFound {len(USER_QUERIES)} queries to evaluate\n")

    # Initialize workflow service
    workflow_service = WorkflowService()

    # Run evaluation for each query
    results = []
    for i, query in enumerate(USER_QUERIES, 1):
        print(f"[{i}/{len(USER_QUERIES)}] ", end="")
        result = await run_single_query(query, workflow_service)
        results.append(result)

    # Calculate average scores
    llm_scored = [r for r in results if r.get("llm_scores", {}).get("accuracy") is not None]

    average_scores = {}
    if llm_scored:
        avg_a = sum(r["llm_scores"]["accuracy"] for r in llm_scored) / len(llm_scored)
        avg_c = sum(r["llm_scores"]["clarity"] for r in llm_scored) / len(llm_scored)

        average_scores = {
            "overall": {
                "accuracy": round(avg_a, 2),
                "clarity": round(avg_c, 2),
                "total_scored": len(llm_scored),
            }
        }

    # Save results
    output_dir = Path("benchmark/results")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "translate_terms_results.json"

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "total_queries": len(USER_QUERIES),
        "results": results,
        "average_scores": average_scores,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("✓ Evaluation complete!")
    print(f"Results saved to: {output_file}")
    print(f"Total queries: {len(USER_QUERIES)}")
    print("=" * 80 + "\n")

    return str(output_file)


def review_results(results_file: str):
    """Display results with LLM scores."""
    print("=" * 80)
    print("Term Translation Results Review")
    print("=" * 80)

    # Load results
    with open(results_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    llm_scored = [r for r in results if r.get("llm_scores", {}).get("accuracy") is not None]

    print(f"\nTotal results: {len(results)}")
    print(f"LLM scored: {len(llm_scored)}\n")

    # Display each result
    for i, result in enumerate(results, 1):
        print("=" * 80)
        print(f"Result {i}/{len(results)}")
        print("=" * 80)
        print(f"\n📋 Query: {result['query']}")
        print(f"\n💬 Response:\n{result['response'][:800]}...")

        # Show LLM scores
        if result.get("llm_scores"):
            llm_scores = result["llm_scores"]
            print("\n🤖 LLM Judge Scores:")
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
        avg_a = sum(r["llm_scores"]["accuracy"] for r in llm_scored) / len(llm_scored)
        avg_c = sum(r["llm_scores"]["clarity"] for r in llm_scored) / len(llm_scored)
        print(f"  Average Accuracy: {avg_a:.2f}")
        print(f"  Average Clarity: {avg_c:.2f}")
        print("=" * 80)


def main():
    """Main entry point."""

    if len(sys.argv) > 1 and sys.argv[1] == "review":
        # Review mode
        results_file = Path("benchmark/results/translate_terms_results.json")
        if not results_file.exists():
            print("No results file found. Run evaluation first.")
            return

        review_results(str(results_file))
    else:
        # Evaluation mode (always uses LLM judge)
        asyncio.run(run_evaluation())


if __name__ == "__main__":
    main()
