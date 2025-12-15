"""Utility functions for the benchmark."""

import re
from enum import Enum
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.config.settings import settings


class Task(Enum):
    """Task enum."""

    MATCHING = "matching"
    SUMMARIZATION = "summarization"
    ELIGIBILITY = "eligibility"
    EXPLANATION = "explanation"
    TRANSLATION = "translation"
    COMPARISON = "comparison"
    MATCHING_COMPARISON = "matching_comparison"


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


class SummarizationEvaluation(BaseModel):
    """Pydantic model for trial summarization evaluation scores."""

    hallucination: int = Field(
        description="Hallucination score (1-5 scale, where 5 is BEST and 1 is WORST). Score whether all information is accurate and grounded in actual trial data."
    )
    hallucination_reasoning: str = Field(description="Brief explanation for the hallucination score")
    accuracy: int = Field(
        description="Accuracy score (1-5 scale, where 5 is BEST and 1 is WORST). Score how accurately the summary captures all key trial information."
    )
    accuracy_reasoning: str = Field(description="Brief explanation for the accuracy score")
    clarity: int = Field(
        description="Clarity score (1-5 scale, where 5 is BEST and 1 is WORST). Score how clear, well-organized, and concise the summary is."
    )
    clarity_reasoning: str = Field(description="Brief explanation for the clarity score")
    language_correction: int = Field(
        description="Language correction score (1-5 scale, where 5 is BEST and 1 is WORST). Score how well the response language matches the user input language."
    )
    language_correction_reasoning: str = Field(
        description="Brief explanation for the language correction score, noting the languages of user input and response"
    )


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


class MatchingComparisonEvaluation(BaseModel):
    """Pydantic model for clinical trial matching comparison evaluation scores."""

    trial_id_count: int = Field(
        description="Number of valid clinical trial IDs (NCT format) extracted from the response. Count actual NCT IDs found."
    )
    trial_ids_extracted: list[str] = Field(
        description="List of all valid clinical trial IDs (NCT format) extracted from the response"
    )
    depth: int = Field(
        description="Depth score (1-5 scale, where 5 is BEST and 1 is WORST). Score the clinical reasoning quality and depth of trial analysis. Higher scores indicate deeper understanding of patient-trial matching beyond surface-level."
    )
    depth_reasoning: str = Field(description="Brief explanation for the depth score")
    relevance: int = Field(
        description="Relevance score (1-5 scale, where 5 is BEST and 1 is WORST). Score how well the response is relevant to the USER INPUT (question or request), regardless of specific trial-patient matching."
    )
    relevance_reasoning: str = Field(description="Brief explanation for the relevance score")
    clarity: int = Field(
        description="Clarity score (1-5 scale, where 5 is BEST and 1 is WORST). Score how clear, well-organized, and easy to understand the response is."
    )
    clarity_reasoning: str = Field(description="Brief explanation for the clarity score")
    completeness: int = Field(
        description="Completeness score (1-5 scale, where 5 is BEST and 1 is WORST). Score how completely the response addresses all key aspects of the patient profile."
    )
    completeness_reasoning: str = Field(description="Brief explanation for the completeness score")


def get_evaluation_model(task: Task) -> BaseModel:
    """Get the evaluation model for the task."""
    if task == Task.MATCHING:
        return MatchingEvaluation
    elif task == Task.SUMMARIZATION:
        return SummarizationEvaluation
    elif task == Task.ELIGIBILITY:
        return EligibilityEvaluation
    elif task == Task.EXPLANATION:
        return ExplanationEvaluation
    elif task == Task.TRANSLATION:
        return TranslationEvaluation
    elif task == Task.COMPARISON:
        return ComparisonEvaluation
    elif task == Task.MATCHING_COMPARISON:
        return MatchingComparisonEvaluation
    else:
        raise ValueError(f"Invalid task: {task}")


def load_llm_judge_prompt(task: Task) -> str:
    """Load the LLM judge prompt template."""
    if task == Task.MATCHING:
        prompt_file = Path("benchmark/prompts/01_llm_judge_prompt.txt")
    elif task == Task.SUMMARIZATION:
        prompt_file = Path("benchmark/prompts/02_llm_judge_summarize_prompt.txt")
    elif task == Task.ELIGIBILITY:
        prompt_file = Path("benchmark/prompts/03_llm_judge_eligibility_prompt.txt")
    elif task == Task.EXPLANATION:
        prompt_file = Path("benchmark/prompts/04_llm_judge_explain_criteria_prompt.txt")
    elif task == Task.TRANSLATION:
        prompt_file = Path("benchmark/prompts/05_llm_judge_translate_terms_prompt.txt")
    elif task == Task.COMPARISON:
        prompt_file = Path("benchmark/prompts/06_llm_judge_compare_trials_prompt.txt")
    elif task == Task.MATCHING_COMPARISON:
        prompt_file = Path("benchmark/prompts/01_llm_judge_matching_comparison_prompt.txt")
    else:
        raise ValueError(f"Invalid task: {task}")
    return prompt_file.read_text(encoding="utf-8")


async def llm_evaluate_response(
    task: Task,
    patient_profile: str = "",
    response: str = "",
    trials_found: list = None,
    user_input: str = "",
    trial_id: str = "",
    trial_title: str = "",
    eligibility_criteria: str = "",
    ground_truth_label: str = "",
    workflow_answer: str = "",
    user_query: str = "",
    trial_information: str = "",
    label_name: str = "",
    ground_truth_explanation: str = "",
) -> dict:
    """Use LLM to evaluate the response quality."""
    # Load and format prompt
    prompt_template = load_llm_judge_prompt(task=task)
    if task == Task.MATCHING:
        # Format trials information
        trials_info = ""
        for i, trial in enumerate(trials_found[:5], 1):  # Top 5 trials
            trials_info += f"\n{i}. {trial['nct_id']}: {trial['title']}\n"
            trials_info += f"   Match Score: {trial['llm_match_score']:.2f}\n"
            trials_info += f"   Reasoning: {trial['match_reasoning']}\n"
        if not trials_info:
            trials_info = "No trials found."

        if not user_input or not patient_profile or not trials_info or not response:
            raise ValueError("Missing required parameters for matching task")
        prompt = prompt_template.format(
            user_input=user_input,
            patient_profile=patient_profile,
            trials_info=trials_info,
            response=response,
        )
    elif task == Task.SUMMARIZATION:
        if not user_input or not trial_id or not response:
            raise ValueError("Missing required parameters for summarization task")
        prompt = prompt_template.format(user_input=user_input, trial_id=trial_id, response=response)
    elif task == Task.ELIGIBILITY:
        if (
            not user_input
            or not patient_profile
            or not trial_id
            or not trial_title
            or not eligibility_criteria
            or not ground_truth_label
            or not label_name
            or not workflow_answer
            or not ground_truth_explanation
        ):
            raise ValueError("Missing required parameters for eligibility task")
        prompt = prompt_template.format(
            user_input=user_input,
            patient_profile=patient_profile,
            trial_id=trial_id,
            trial_title=trial_title,
            eligibility_criteria=eligibility_criteria,
            ground_truth_label=f"{ground_truth_label} ({label_name})",
            ground_truth_explanation=ground_truth_explanation,
            workflow_answer=workflow_answer,
        )
    elif task == Task.EXPLANATION:
        if not user_input or not trial_id or not trial_title or not eligibility_criteria or not response:
            raise ValueError("Missing required parameters for explanation task")
        prompt = prompt_template.format(
            user_input=user_input,
            trial_id=trial_id,
            trial_title=trial_title,
            eligibility_criteria=eligibility_criteria,
            response=response,
        )
    elif task == Task.TRANSLATION:
        if not user_query or not response:
            raise ValueError("Missing required parameters for translation task")
        prompt = prompt_template.format(user_query=user_query, response=response)
    elif task == Task.COMPARISON:
        if not user_input or not trial_information or not response:
            raise ValueError("Missing required parameters for comparison task")
        prompt = prompt_template.format(user_input=user_input, trial_information=trial_information, response=response)
    elif task == Task.MATCHING_COMPARISON:
        if not user_input or not response:
            raise ValueError("Missing required parameters for matching comparison task")
        prompt = prompt_template.format(
            user_input=user_input,
            response=response,
        )
    else:
        raise ValueError(f"Invalid task: {task}")

    # Initialize the model with structured output
    llm = ChatOpenAI(model=settings.llm_judge_model, temperature=0.0)
    structured_llm = llm.with_structured_output(get_evaluation_model(task))

    try:
        # Get structured output
        evaluation_result = await structured_llm.ainvoke([HumanMessage(content=prompt)])

        # Build result dict conditionally based on task
        if task == Task.MATCHING_COMPARISON:
            # Special handling for matching comparison evaluation
            result = {
                "trial_id_count": evaluation_result.trial_id_count,
                "trial_ids_extracted": evaluation_result.trial_ids_extracted,
                "depth": evaluation_result.depth,
                "depth_reasoning": evaluation_result.depth_reasoning,
                "relevance": evaluation_result.relevance,
                "relevance_reasoning": evaluation_result.relevance_reasoning,
                "clarity": evaluation_result.clarity,
                "clarity_reasoning": evaluation_result.clarity_reasoning,
                "completeness": evaluation_result.completeness,
                "completeness_reasoning": evaluation_result.completeness_reasoning,
            }
        else:
            result = {
                "accuracy": evaluation_result.accuracy,
                "accuracy_reasoning": evaluation_result.accuracy_reasoning,
                "clarity": evaluation_result.clarity,
                "clarity_reasoning": evaluation_result.clarity_reasoning,
                "language_correction": evaluation_result.language_correction,
                "language_correction_reasoning": evaluation_result.language_correction_reasoning,
            }

            # Only include hallucination if the evaluation model has it
            if hasattr(evaluation_result, "hallucination"):
                result["hallucination"] = evaluation_result.hallucination
                result["hallucination_reasoning"] = evaluation_result.hallucination_reasoning

        return result
    except Exception as e:
        print(f"  LLM evaluation error: {str(e)}")
        result = {
            "accuracy": None,
            "accuracy_reasoning": f"Error: {str(e)}",
            "clarity": None,
            "clarity_reasoning": f"Error: {str(e)}",
            "language_correction": None,
            "language_correction_reasoning": f"Error: {str(e)}",
        }

        # Only include hallucination in error response if task supports it
        if task != Task.TRANSLATION and task != Task.MATCHING_COMPARISON:
            result["hallucination"] = None
            result["hallucination_reasoning"] = f"Error: {str(e)}"
        elif task == Task.MATCHING_COMPARISON:
            # Add matching comparison specific error fields
            result["trial_id_count"] = 0
            result["trial_ids_extracted"] = []
            result["depth"] = None
            result["depth_reasoning"] = f"Error: {str(e)}"
            result["relevance"] = None
            result["relevance_reasoning"] = f"Error: {str(e)}"
            result["completeness"] = None
            result["completeness_reasoning"] = f"Error: {str(e)}"

        return result


def extract_trial_ids_from_text(text: str) -> list[str]:
    """
    Extract clinical trial IDs (NCT format) from text using regex pattern.

    Args:
        text: Text response that may contain trial IDs

    Returns:
        List of unique trial IDs found (NCT format, uppercase)
    """
    # Pattern for NCT IDs: NCT followed by 8 digits
    pattern = r"NCT\d{8}"
    matches = re.findall(pattern, text, re.IGNORECASE)
    # Normalize to uppercase and remove duplicates while preserving order
    unique_ids = []
    seen = set()
    for match in matches:
        upper_match = match.upper()
        if upper_match not in seen:
            seen.add(upper_match)
            unique_ids.append(upper_match)
    return unique_ids


async def extract_trial_ids_with_llm(text: str) -> list[str]:
    """
    Extract clinical trial IDs from text using LLM for more robust extraction.
    Falls back to regex if LLM extraction fails.

    Args:
        text: Text response that may contain trial IDs

    Returns:
        List of unique trial IDs found (NCT format, uppercase)
    """
    # First try regex extraction
    regex_ids = extract_trial_ids_from_text(text)

    # If we found IDs with regex, return them
    if regex_ids:
        return regex_ids

    # Otherwise, try LLM extraction for edge cases
    try:
        extraction_prompt = f"""Extract all clinical trial IDs (NCT format: NCT followed by 8 digits) from the following text.
Return only the trial IDs, one per line, in the format NCT########.
If no trial IDs are found, return "NONE".

Text:
{text}

Trial IDs:"""

        llm = ChatOpenAI(model=settings.llm_model, temperature=0.0)
        response = await llm.ainvoke([HumanMessage(content=extraction_prompt)])
        response_text = response.content.strip()

        if response_text.upper() == "NONE" or not response_text:
            return []

        # Extract IDs from LLM response
        ids = extract_trial_ids_from_text(response_text)
        return ids
    except Exception:
        # Fall back to regex if LLM fails
        return regex_ids


if __name__ == "__main__":
    print("hello")
