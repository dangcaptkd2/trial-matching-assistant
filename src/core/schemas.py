"""Pydantic schemas for structured LLM outputs in the clinical trial assistant workflow."""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class IntentType(str, Enum):
    """Enumeration of possible intent types."""

    GREETING = "GREETING"
    OFF_TOPIC = "OFF_TOPIC"
    FIND_TRIALS = "FIND_TRIALS"
    SUMMARIZE_TRIAL = "SUMMARIZE_TRIAL"
    CHECK_ELIGIBILITY = "CHECK_ELIGIBILITY"
    EXPLAIN_CRITERIA = "EXPLAIN_CRITERIA"
    COMPARE_TRIALS = "COMPARE_TRIALS"
    TRANSLATE_TERMS = "TRANSLATE_TERMS"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"


class IntentClassification(BaseModel):
    """Schema for intent classification output from LLM.

    Used by: intent_classification_prompt.txt
    """

    intent_type: IntentType = Field(
        description="The classified intent type: GREETING, OFF_TOPIC, FIND_TRIALS, SUMMARIZE_TRIAL, CHECK_ELIGIBILITY, EXPLAIN_CRITERIA, COMPARE_TRIALS, TRANSLATE_TERMS, or NEEDS_CLARIFICATION"
    )
    patient_info: Optional[str] = Field(
        default=None,
        description="Extracted patient information as a single string (age, gender, condition, symptoms, diagnosis, treatment). Must be in English even if user input is in another language. Return null if no patient info provided.",
    )
    trial_ids: Optional[List[str]] = Field(
        default=None,
        description="List of extracted trial IDs (format: NCT followed by 8 digits). Return as list of strings, or null if none found.",
    )
    clarification_reason: str = Field(
        default="",
        description="Brief reason if NEEDS_CLARIFICATION, else empty string",
    )


class RerankScore(BaseModel):
    """Schema for trial reranking score output from LLM.

    Used by: rerank_prompt.txt
    """

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Match score between 0.0 and 1.0 where 1.0 = Patient clearly meets ALL inclusion criteria and does NOT meet any exclusion criteria, 0.8-0.9 = Patient likely eligible with minor uncertainties, 0.5-0.7 = Some criteria match but significant uncertainties or potential exclusions, 0.0-0.4 = Patient likely does NOT meet key criteria",
    )
    reasoning: str = Field(description="Brief explanation of why this score was assigned")
