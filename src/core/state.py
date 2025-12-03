"""State definition for the clinical trial search and rerank workflow."""

from typing import TypedDict
from langchain_core.messages import BaseMessage


class GraphState(TypedDict):
    """State for the clinical trial search and rerank workflow"""

    user_input: str  # Original user query
    messages: list[BaseMessage]  # Conversation history
    query_type: (
        str  # Query classification: CHITCHAT, FIND_TRIALS, SUMMARIZE_TRIAL, CLARIFY
    )
    trial_ids: list  # List of trial IDs to summarize (e.g., ["NCT12345678"])
    trial_search_query: (
        str  # Search query for finding trials (used in CLARIFY for trial_id)
    )
    search_query: str  # Improved search query for ES (FIND_TRIALS)
    patient_profile: str  # Extracted/improved patient profile
    chitchat_response: str  # Response for non-trial questions
    clarification_type: str  # Type: "trial_id", "patient_profile", etc.
    clarification_context: str  # Context about what user asked for
    clarification_search_results: list  # Search results if applicable
    search_results: list  # Results from ES search
    reranked_results: list  # Results after LLM reranking
    final_answer: str  # Synthesized answer for the user
    top_k: int  # Number of results to retrieve (default 10)
