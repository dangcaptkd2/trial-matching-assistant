#!/usr/bin/env python3
"""Evaluation script for patient matching feature using 50 TREC topics."""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from benchmark.xml_data_utils import get_topics_xml_file
from src.api.services.workflow import WorkflowService
from src.config.settings import settings


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


def load_llm_judge_prompt() -> str:
    """Load the LLM judge prompt template."""
    prompt_file = Path("benchmark/prompts/llm_judge_prompt.txt")
    return prompt_file.read_text(encoding="utf-8")


async def llm_evaluate_response(patient_profile: str, response: str, trials_found: list) -> dict:
    """Use LLM to evaluate the response quality."""
    # Format trials information
    trials_info = ""
    for i, trial in enumerate(trials_found[:5], 1):  # Top 5 trials
        trials_info += f"\n{i}. {trial['nct_id']}: {trial['title']}\n"
        trials_info += f"   Match Score: {trial['llm_match_score']:.2f}\n"
        trials_info += f"   Reasoning: {trial['match_reasoning']}\n"

    if not trials_info:
        trials_info = "No trials found."

    # Load and format prompt
    prompt_template = load_llm_judge_prompt()
    prompt = prompt_template.format(
        patient_profile=patient_profile,
        trials_info=trials_info,
        response=response,
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


async def run_single_topic(topic: dict, workflow_service: WorkflowService) -> dict:
    """Run workflow for a single topic and capture results."""
    topic_number = topic["number"]
    patient_profile = topic["content"]
    query = "find trials for this patient: " + patient_profile

    print(f"Running topic {topic_number}...", end=" ", flush=True)

    start_time = datetime.now()
    result_data = None

    try:
        async for event in workflow_service.invoke_workflow(
            user_input=query,
            thread_id=f"eval-topic-{topic_number}",
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
        }

        # Always run LLM evaluation
        if result_data:
            llm_scores = await llm_evaluate_response(patient_profile, response, trials)
            evaluation_result["llm_scores"] = llm_scores
            print(
                f"✓ ({elapsed:.1f}s) [LLM: H={llm_scores.get('hallucination', '?')} A={llm_scores.get('accuracy', '?')} C={llm_scores.get('clarity', '?')}]"
            )
            evaluation_result["llm_scores"] = llm_scores
        else:
            print(f"✓ ({elapsed:.1f}s)")

        return evaluation_result

    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"✗ Error: {str(e)}")
        return {
            "topic_number": topic_number,
            "patient_profile": patient_profile,
            "response": f"ERROR: {str(e)}",
            "trials_found": [],
            "execution_time": elapsed,
            "error": str(e),
        }


async def run_evaluation(topics_file: str = "data/topics2022.xml"):
    """Run evaluation on all topics with LLM-as-Judge."""
    print("=" * 80)
    print("Clinical Trial Matching Evaluation")
    print("(with LLM-as-Judge automatic scoring)")
    print("=" * 80)

    # Load topics
    print(f"\nLoading topics from {topics_file}...")
    topics_data = get_topics_xml_file(topics_file)
    topics = topics_data.get("topics", [])[:10]

    print(f"Found {len(topics)} topics to evaluate\n")

    # Initialize workflow service
    workflow_service = WorkflowService()

    # Run evaluation for each topic
    results = []
    for i, topic in enumerate(topics, 1):
        print(f"[{i}/{len(topics)}] ", end="")
        result = await run_single_topic(topic, workflow_service)
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

    output_file = output_dir / "matching_results.json"

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "topics_file": topics_file,
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
    print("=" * 80 + "\n")

    return str(output_file)


def review_results(results_file: str):
    """Display results with LLM scores."""
    print("=" * 80)
    print("Evaluation Results Review")
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
        print(f"\n📋 Patient Profile:\n{result['patient_profile'][:300]}...")
        print(f"\n💬 Response:\n{result['response'][:500]}...")
        print(f"\n🔬 Trials Found: {len(result['trials_found'])}")

        if result["trials_found"]:
            print("\nTop 3 trials:")
            for j, trial in enumerate(result["trials_found"][:3], 1):
                print(f"  {j}. {trial['nct_id']}: {trial['title'][:60]}...")
                print(f"     Score: {trial['llm_match_score']:.2f}")

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
        if len(sys.argv) > 2:
            results_file = sys.argv[2]
        else:
            # Find latest results file
            results_dir = Path("benchmark/results")
            if not results_dir.exists():
                print("No results directory found. Run evaluation first.")
                return

            results_file = str(results_dir / "matching_results.json")
            if not Path(results_file).exists():
                print("No results file found. Run evaluation first.")
                return

            print(f"Using results: {results_file}\n")

        review_results(results_file)
    else:
        # Evaluation mode (always uses LLM judge)
        asyncio.run(run_evaluation())


if __name__ == "__main__":
    main()
