#!/usr/bin/env python3
"""Evaluation script for patient matching feature using 50 TREC topics."""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

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


class MatchingEvaluation(BaseModel):
    """Pydantic model for trial matching evaluation scores."""

    hallucination: int = Field(
        description="Hallucination score (1-5 scale, where 5 is BEST and 1 is WORST). Score whether all information is grounded in the trial data."
    )
    hallucination_reasoning: str = Field(description="Brief explanation for the hallucination score")
    accuracy: int = Field(
        description="Accuracy score (1-5 scale, where 5 is BEST and 1 is WORST). Score how relevant and well-matched the trials are to patient criteria."
    )
    accuracy_reasoning: str = Field(description="Brief explanation for the accuracy score")
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
    """Load the LLM judge prompt template."""
    prompt_file = Path("benchmark/prompts/01_llm_judge_prompt.txt")
    return prompt_file.read_text(encoding="utf-8")


async def llm_evaluate_response(patient_profile: str, response: str, trials_found: list, user_input: str = "") -> dict:
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
        user_input=user_input,
        patient_profile=patient_profile,
        trials_info=trials_info,
        response=response,
    )

    # Initialize the model with structured output
    llm = ChatOpenAI(model=settings.llm_judge_model, temperature=0.0)
    structured_llm = llm.with_structured_output(MatchingEvaluation)

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

    try:
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
            llm_scores = await llm_evaluate_response(patient_profile, response, trials, user_input=query)
            evaluation_result["llm_scores"] = llm_scores
            print(
                f"✓ ({elapsed:.1f}s) [LLM: H={llm_scores.get('hallucination', '?')} A={llm_scores.get('accuracy', '?')} C={llm_scores.get('clarity', '?')} L={llm_scores.get('language_correction', '?')}]"
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
            "language": language,
        }


def load_translated_topics(translated_file: str = "benchmark/results/translated_topics.json") -> dict:
    """Load translated topics from JSON file."""
    try:
        with open(translated_file, "r", encoding="utf-8") as f:
            translated_data = json.load(f)
        # Convert to dict by topic number for easy lookup
        translated_dict = {item["number"]: item for item in translated_data}
        return translated_dict
    except FileNotFoundError:
        print(f"Warning: Translated topics file not found: {translated_file}")
        return {}
    except Exception as e:
        print(f"Warning: Error loading translated topics: {str(e)}")
        return {}


async def run_evaluation(topics_file: str = "data/topics2022.xml", language: str = "en"):
    """Run evaluation on all topics with LLM-as-Judge."""
    print("=" * 80)
    print("Clinical Trial Matching Evaluation")
    print(f"Language: {language.upper()}")
    print("(with LLM-as-Judge automatic scoring)")
    print("=" * 80)

    # Load topics
    print(f"\nLoading topics from {topics_file}...")
    topics_data = get_topics_xml_file(topics_file)
    topics = topics_data.get("topics", [])[:10]

    # Load translated topics if Vietnamese
    translated_topics = {}
    if language == "vi":
        print("Loading Vietnamese translations...")
        translated_topics = load_translated_topics()
        if not translated_topics:
            print("Warning: No translated topics found. Falling back to English.")
            language = "en"
        else:
            print(f"✓ Loaded {len(translated_topics)} translated topics")

    # Merge translated content into topics
    if language == "vi" and translated_topics:
        for topic in topics:
            topic_number = topic["number"]
            if topic_number in translated_topics:
                topic["vi_content"] = translated_topics[topic_number]["vi_content"]

    print(f"Found {len(topics)} topics to evaluate\n")

    # Initialize workflow service
    workflow_service = WorkflowService()

    # Run evaluation for each topic
    results = []
    for i, topic in enumerate(topics, 1):
        print(f"[{i}/{len(topics)}] ", end="")
        result = await run_single_topic(topic, workflow_service, language=language)
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
                "language_correction": round(avg_l, 2) if avg_l > 0 else None,
                "total_scored": len(llm_scored),
            }
        }

    # Save results
    output_dir = Path("benchmark/results")
    output_dir.mkdir(exist_ok=True)

    # Include language in filename to avoid overwriting
    output_file = output_dir / f"matching_results_{language}.json"

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "topics_file": topics_file,
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

    topics_file = "data/topics2022.xml"
    # Evaluation mode
    asyncio.run(run_evaluation(topics_file=topics_file, language=args.lang))


if __name__ == "__main__":
    main()
