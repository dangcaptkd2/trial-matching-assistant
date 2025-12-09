"""State definition for the clinical trial search and rerank workflow."""

from typing import TypedDict

from langchain_core.messages import BaseMessage


class GraphState(TypedDict):
    """State for the clinical trial search and rerank workflow"""

    user_input: str  # Original user query
    messages: list[BaseMessage]  # Conversation history

    # Intent classification fields
    intent_type: (
        str  # Intent: GREETING, OFF_TOPIC, FIND_TRIALS, SUMMARIZE_TRIAL, CHECK_ELIGIBILITY, NEEDS_CLARIFICATION
    )
    patient_info: str  # Extracted patient information (string or None)
    trial_ids: list  # List of extracted trial IDs (e.g., ["NCT12345678"] or None)
    clarification_reason: str  # Minimal context for clarification node

    # Trial and search fields
    trial_data: list  # Fetched trial documents from Elasticsearch (list of dicts)
    trial_search_query: str  # Search query for finding trials (used in CLARIFY for trial_id)
    chitchat_response: str  # Response for non-trial questions

    # Search and results fields
    search_results: list  # Results from ES search
    reranked_results: list  # Results after LLM reranking
    final_answer: str  # Synthesized answer for the user
    top_k: int  # Number of results to retrieve (default 10)
