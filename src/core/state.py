"""State definition for the clinical trial search and rerank workflow."""

from typing import TypedDict
from langchain_core.messages import BaseMessage


class GraphState(TypedDict):
    """State for the clinical trial search and rerank workflow"""

    user_input: str  # Original user query
    messages: list[BaseMessage]  # Conversation history
    needs_trial_search: bool  # Whether query requires trial search
    needs_trial_lookup: bool  # Whether query requires trial ID lookup
    trial_ids: list  # List of trial IDs to lookup (e.g., ["NCT12345678"])
    search_query: str  # Improved search query for ES
    patient_profile: str  # Extracted/improved patient profile
    chitchat_response: str  # Response for non-trial questions
    trial_lookup_results: list  # Results from trial ID lookup
    search_results: list  # Results from ES search
    reranked_results: list  # Results after LLM reranking
    final_answer: str  # Synthesized answer for the user
    top_k: int  # Number of results to retrieve (default 10)
