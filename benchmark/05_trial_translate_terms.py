#!/usr/bin/env python3
"""Evaluation script for translate terms feature using a list of user queries."""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.api.services.workflow import WorkflowService
from src.config.settings import settings

# List of user queries to evaluate (edit this list manually)
USER_QUERIES_EN = [
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

USER_QUERIES_VI = [
    "'Progression-free survival' có nghĩa là gì?",
    "Giải thích đột biến EGFR",
    "Các thuật ngữ này có nghĩa là gì: liệu pháp bổ trợ và di căn?",
    "Hóa trị là gì?",
    "Giải thích 'overall survival' có nghĩa là gì",
    "'Biopsy' có nghĩa là gì trong chẩn đoán ung thư?",
    "Bạn có thể giải thích 'tumor marker' có nghĩa là gì không?",
    "Ý nghĩa của 'immunotherapy' là gì?",
    "'Staging' có nghĩa là gì trong ung thư?",
    "Giải thích 'remission' có nghĩa là gì trong thuật ngữ y tế",
]


def format_response(result: dict) -> str:
    """Extract the response from workflow result."""
    if result.get("chitchat_response"):
        return result.get("chitchat_response", "")
    elif result.get("final_answer"):
        return result.get("final_answer", "")
    else:
        return "No response generated."


class TranslationEvaluation(BaseModel):
    """Pydantic model for translation evaluation scores."""

    accuracy: int = Field(
        description="Accuracy score (1-5 scale, where 5 is BEST and 1 is WORST). Score how accurately the explanation captures the meaning of all medical terms mentioned."
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
    """Load the LLM judge prompt template for translate terms evaluation."""
    prompt_file = Path("benchmark/prompts/05_llm_judge_translate_terms_prompt.txt")
    return prompt_file.read_text(encoding="utf-8")


async def llm_evaluate_translation(user_query: str, response: str) -> dict:
    """Use LLM to evaluate the translation quality."""
    # Load and format prompt
    prompt_template = load_llm_judge_prompt()
    prompt = prompt_template.format(user_query=user_query, response=response)

    # Initialize the model with structured output
    llm = ChatOpenAI(model=settings.llm_judge_model, temperature=0.0)
    structured_llm = llm.with_structured_output(TranslationEvaluation)

    try:
        # Get structured output
        evaluation_result = await structured_llm.ainvoke([HumanMessage(content=prompt)])

        return {
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
            "accuracy": None,
            "accuracy_reasoning": f"Error: {str(e)}",
            "clarity": None,
            "clarity_reasoning": f"Error: {str(e)}",
            "language_correction": None,
            "language_correction_reasoning": f"Error: {str(e)}",
        }


async def run_single_query(query: str, workflow_service: WorkflowService, language: str = "en") -> dict:
    """Run workflow for a single query and capture results."""
    print(f"Running query: {query[:60]}...", end=" ", flush=True)

    start_time = datetime.now()
    result_data = None

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
        "language": language,
    }

    # Always run LLM evaluation
    if result_data:
        llm_scores = await llm_evaluate_translation(query, response)
        evaluation_result["llm_scores"] = llm_scores
        print(
            f"✓ ({elapsed:.1f}s) [LLM: A={llm_scores.get('accuracy', '?')} C={llm_scores.get('clarity', '?')} L={llm_scores.get('language_correction', '?')}]"
        )
    else:
        print(f"✓ ({elapsed:.1f}s)")

    return evaluation_result


async def run_evaluation(language: str = "en"):
    """Run evaluation on all user queries."""
    print("=" * 80)
    print("Medical Terms Translation Evaluation")
    print(f"Language: {language.upper()}")
    print("(with LLM-as-Judge automatic scoring)")
    print("=" * 80)

    # Select queries based on language
    if language == "vi":
        user_queries = USER_QUERIES_VI
    else:
        user_queries = USER_QUERIES_EN

    if not user_queries:
        print(f"\n❌ Error: No user queries provided for language '{language}'!")
        print("Please add queries to the USER_QUERIES list in the script.")
        return

    print(f"\nFound {len(user_queries)} queries to evaluate\n")

    # Initialize workflow service
    workflow_service = WorkflowService()

    # Run evaluation for each query
    results = []
    for i, query in enumerate(user_queries, 1):
        print(f"[{i}/{len(user_queries)}] ", end="")
        result = await run_single_query(query, workflow_service, language=language)
        results.append(result)

    # Calculate average scores
    llm_scored = [r for r in results if r.get("llm_scores", {}).get("accuracy") is not None]

    average_scores = {}
    if llm_scored:
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
    output_file = output_dir / f"translate_terms_results_{language}.json"

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "language": language,
        "total_queries": len(user_queries),
        "results": results,
        "average_scores": average_scores,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("✓ Evaluation complete!")
    print(f"Results saved to: {output_file}")
    print(f"Total queries: {len(user_queries)}")
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
        avg_a = sum(r["llm_scores"]["accuracy"] for r in llm_scored) / len(llm_scored)
        avg_c = sum(r["llm_scores"]["clarity"] for r in llm_scored) / len(llm_scored)
        lang_corr_scored = [r for r in llm_scored if r["llm_scores"].get("language_correction") is not None]
        avg_l = (
            sum(r["llm_scores"]["language_correction"] for r in lang_corr_scored) / len(lang_corr_scored)
            if lang_corr_scored
            else None
        )
        print(f"  Average Accuracy: {avg_a:.2f}")
        print(f"  Average Clarity: {avg_c:.2f}")
        if avg_l is not None:
            print(f"  Average Language Correction: {avg_l:.2f}")
        print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Medical Terms Translation Evaluation Script",
    )

    parser.add_argument(
        "--lang",
        "-l",
        choices=["en", "vi"],
        default="en",
        help="Language for evaluation: 'en' for English, 'vi' for Vietnamese",
    )

    parser.add_argument(
        "mode",
        nargs="?",
        choices=["review"],
        help="Mode: 'review' to review existing results, or omit to run evaluation",
    )

    args = parser.parse_args()

    if args.mode == "review":
        # Review mode
        results_file = Path(f"benchmark/results/translate_terms_results_{args.lang}.json")
        if not results_file.exists():
            print(f"No results file found for language '{args.lang}'. Run evaluation first.")
            return
        review_results(str(results_file))
    else:
        # Evaluation mode (always uses LLM judge)
        asyncio.run(run_evaluation(language=args.lang))


if __name__ == "__main__":
    main()
