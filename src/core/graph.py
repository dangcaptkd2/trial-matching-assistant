"""LangGraph workflow definition and nodes for clinical trial matching."""

import asyncio
import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END

from src.core.state import GraphState
from src.core.memory import get_checkpointer
from src.services.search import ElasticsearchTrialSearcher
from src.prompts.prompts import (
    rerank_prompt,
    synthesis_prompt,
    reception_prompt,
    trial_lookup_synthesis_prompt,
)
from src.config.settings import settings


# Initialize ES searcher
es_searcher = ElasticsearchTrialSearcher(index_name=settings.es_index_name)


# ————— MEMORY HELPERS —————
def needs_conversation_context(user_input: str, has_history: bool) -> bool:
    """
    Rule-based check if conversation context is needed.
    Uses heuristics to detect follow-up questions or references.
    """
    if not has_history:
        return False

    user_lower = user_input.lower()

    # Strong indicators that context IS needed
    reference_indicators = [
        "that trial",
        "those trials",
        "the trial",
        "it",
        "them",
        "tell me more",
        "what about",
        "remind me",
        "again",
        "previous",
        "earlier",
    ]

    # Check for pronouns/references
    has_reference = any(indicator in user_lower for indicator in reference_indicators)

    # Very short queries might be follow-ups
    is_short_query = len(user_input.split()) <= 4

    return has_reference or is_short_query


def get_conversation_context(
    state: GraphState, max_messages: int = 3
) -> tuple[str, str]:
    """
    Get conversation context using hybrid approach (rules decide).

    Returns:
        (context: str, instruction: str)
    """
    messages = state.get("messages", [])
    user_input = state.get("user_input", "")

    if not messages:
        return "", ""

    # Rule-based decision (fast, no LLM call)
    if not needs_conversation_context(user_input, has_history=True):
        return "", ""

    # Get recent messages
    recent = messages[-max_messages:]
    context_lines = []
    for msg in recent:
        role = "User" if msg.type == "human" else "Assistant"
        content = msg.content[:200] if hasattr(msg, "content") else str(msg)[:200]
        context_lines.append(f"{role}: {content}")

    context = "\n".join(context_lines)
    instruction = "Use the conversation context above if the current query references previous messages (e.g., pronouns, 'that trial', 'tell me more')."

    return context, instruction


def reception_node(state: GraphState) -> GraphState:
    """Node: Reception - Classify user input and route accordingly"""
    user_input = state.get("user_input", "")
    messages = state.get("messages", [])

    if not user_input:
        return {
            "needs_trial_search": False,
            "needs_trial_lookup": False,
            "trial_ids": [],
            "chitchat_response": "I'm here to help you find clinical trials. Please tell me about your condition or what you're looking for.",
        }

    # Get conversation context using hybrid approach
    context, context_instruction = get_conversation_context(state, max_messages=3)

    # Create LLM instance for reception
    reception_llm = ChatOpenAI(
        model=settings.llm_model, temperature=settings.temperature
    )

    # Format prompt with context
    conversation_context_section = (
        f"\nPREVIOUS CONVERSATION:\n{context}\n" if context else ""
    )

    prompt = reception_prompt.format(
        user_input=user_input,
        conversation_context=conversation_context_section,
        context_instruction=context_instruction,
    )

    # Use config to identify this node in LangSmith
    config = RunnableConfig(
        metadata={"node": "reception"},
        tags=["reception"],
        run_name="reception",
    )

    response = reception_llm.invoke([HumanMessage(content=prompt)], config=config)
    content = response.content.strip()

    # Extract JSON from response
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    try:
        result = json.loads(content)

        # Update messages with user input and response
        updated_messages = list(messages) if messages else []
        updated_messages.append(HumanMessage(content=user_input))

        # Add assistant response if chitchat
        if result.get("chitchat_response"):
            updated_messages.append(
                AIMessage(content=result.get("chitchat_response", ""))
            )

        return {
            "messages": updated_messages,
            "needs_trial_search": bool(result.get("needs_trial_search", False)),
            "needs_trial_lookup": bool(result.get("needs_trial_lookup", False)),
            "trial_ids": result.get("trial_ids", []),
            "chitchat_response": result.get("chitchat_response", ""),
            "search_query": result.get("search_query", ""),
            "patient_profile": result.get("patient_profile", ""),
        }
    except json.JSONDecodeError:
        # Fallback: try to detect trial IDs
        import re

        # Update messages with user input
        updated_messages = list(messages) if messages else []
        updated_messages.append(HumanMessage(content=user_input))

        trial_ids = re.findall(r"NCT\d{8}", user_input)
        if trial_ids:
            return {
                "messages": updated_messages,
                "needs_trial_search": False,
                "needs_trial_lookup": True,
                "trial_ids": trial_ids,
                "chitchat_response": "",
                "search_query": "",
                "patient_profile": "",
            }
        # Otherwise assume patient matching
        return {
            "messages": updated_messages,
            "needs_trial_search": True,
            "needs_trial_lookup": False,
            "trial_ids": [],
            "chitchat_response": "",
            "search_query": user_input,
            "patient_profile": user_input,
        }


def route_query(state: GraphState) -> str:
    """Conditional edge function: route based on query type"""
    if state.get("needs_trial_lookup", False):
        return "lookup_trials"
    elif state.get("needs_trial_search", False):
        return "search"
    else:
        return "chitchat_response"


async def search_clinical_trials_node(state: GraphState) -> GraphState:
    """Node: Search clinical trials using Elasticsearch"""
    query = state.get("search_query", "")
    patient_profile = state.get("patient_profile", "")
    top_k = state.get("top_k", 10)

    # Use patient_profile if available, otherwise use search_query
    search_text = patient_profile if patient_profile else query

    # Directly call the async function - no need for event loop juggling!
    search_results = await es_searcher.search_with_full_documents(
        search_text, top_k=top_k
    )

    # Format results with eligibility criteria
    formatted_results = []
    for result in search_results:
        source = result.get("source", {})
        nct_id = source.get("nct_id") or source.get("id") or result.get("id", "N/A")
        title = (
            source.get("brief_title")
            or source.get("official_title")
            or source.get("text", "")[:200] + "..."
        )
        eligibility = source.get("eligibility_criteria", "")

        formatted_results.append(
            {
                "nct_id": nct_id,
                "title": title,
                "eligibility_criteria": eligibility,
                "es_score": result.get("score", 0.0),
            }
        )
    return {"search_results": formatted_results}


async def rerank_with_llm_node(state: GraphState) -> GraphState:
    """Node: Rerank search results using LLM as cross-encoder"""
    search_results = state.get("search_results", [])
    patient_profile = state.get("patient_profile", "")

    if not search_results:
        return {"reranked_results": []}

    # Create LLM instance for cross-encoding
    cross_encoder_llm = ChatOpenAI(
        model=settings.llm_model, temperature=settings.temperature
    )

    async def score_trial(trial: dict) -> dict:
        """Use LLM to score a single trial against patient profile"""
        eligibility = trial.get("eligibility_criteria", "")
        if not eligibility:
            return {
                **trial,
                "llm_score": 0.0,
                "match_reasoning": "No eligibility criteria available",
            }

        prompt = rerank_prompt.format(
            patient_profile=patient_profile, eligibility=eligibility
        )

        try:
            config = RunnableConfig(
                metadata={"node": "rerank_with_llm", "operation": "trial_scoring"},
                tags=["rerank", "cross-encoder", "trial-matching"],
                run_name="rerank_trial_scoring",
            )
            response = await cross_encoder_llm.ainvoke(
                [HumanMessage(content=prompt)], config=config
            )
            content = response.content.strip()

            # Extract JSON from response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            result = json.loads(content)
            return {
                **trial,
                "llm_score": float(result.get("score", 0.0)),
                "match_reasoning": result.get("reasoning", "No reasoning provided"),
            }
        except Exception as e:
            return {
                **trial,
                "llm_score": 0.0,
                "match_reasoning": f"Error in LLM evaluation: {str(e)}",
            }

    # Run all LLM calls in parallel - much simpler with async node!
    tasks = [score_trial(trial) for trial in search_results]
    scored_results = await asyncio.gather(*tasks)

    # Sort by LLM score (descending)
    scored_results.sort(key=lambda x: x.get("llm_score", 0.0), reverse=True)

    # Format final results
    reranked_results = []
    for result in scored_results:
        reranked_results.append(
            {
                "nct_id": result.get("nct_id", "N/A"),
                "title": result.get("title", "N/A"),
                "es_score": result.get("es_score", 0.0),
                "llm_match_score": result.get("llm_score", 0.0),
                "match_reasoning": result.get("match_reasoning", "N/A"),
            }
        )

    return {"reranked_results": reranked_results}


def synthesize_answer_node(state: GraphState) -> GraphState:
    """Node: Synthesize reranked results into a natural language answer"""
    reranked_results = state.get("reranked_results", [])
    patient_profile = state.get("patient_profile", "")
    messages = state.get("messages", [])

    if not reranked_results:
        final_answer = "I'm sorry, but I couldn't find any clinical trials matching your profile. Please try adjusting your search criteria or consult with your healthcare provider for alternative options."
        # Update messages
        updated_messages = list(messages) if messages else []
        updated_messages.append(AIMessage(content=final_answer))
        return {
            "final_answer": final_answer,
            "messages": updated_messages,
        }

    # Format reranked trials for the prompt
    trials_text = ""
    for i, trial in enumerate(reranked_results[:5], 1):  # Top 5 trials
        trials_text += f"\n{i}. {trial.get('title', 'N/A')}\n"
        trials_text += f"   NCT ID: {trial.get('nct_id', 'N/A')}\n"
        trials_text += f"   Match Score: {trial.get('llm_match_score', 0.0):.2f}\n"
        trials_text += f"   Reasoning: {trial.get('match_reasoning', 'N/A')}\n"

    # Get conversation context
    context, context_instruction = get_conversation_context(state, max_messages=3)
    conversation_context_section = (
        f"\nPREVIOUS CONVERSATION:\n{context}\n" if context else ""
    )

    # Create LLM instance for synthesis
    synthesis_llm = ChatOpenAI(
        model=settings.llm_model, temperature=settings.temperature
    )

    prompt = synthesis_prompt.format(
        patient_profile=patient_profile,
        reranked_trials=trials_text,
        conversation_context=conversation_context_section,
        context_instruction=context_instruction,
    )

    # Use synchronous invoke instead of async to avoid event loop issues
    config = RunnableConfig(
        metadata={"node": "synthesize_answer", "operation": "answer_synthesis"},
        tags=["synthesize", "answer-generation"],
        run_name="synthesize_answer",
    )
    response = synthesis_llm.invoke([HumanMessage(content=prompt)], config=config)
    final_answer = response.content.strip()

    # Update messages with final answer
    updated_messages = list(messages) if messages else []
    updated_messages.append(AIMessage(content=final_answer))

    return {
        "final_answer": final_answer,
        "messages": updated_messages,
        "synthesis_prompt": prompt,
    }


def lookup_trials_by_id_node(state: GraphState) -> GraphState:
    """Node: Lookup specific trials by their IDs"""
    trial_ids = state.get("trial_ids", [])
    messages = state.get("messages", [])

    if not trial_ids:
        final_answer = "I couldn't find any trial IDs in your query. Please provide trial IDs in the format NCT12345678."
        # Update messages
        updated_messages = list(messages) if messages else []
        updated_messages.append(AIMessage(content=final_answer))
        return {
            "trial_lookup_results": [],
            "final_answer": final_answer,
            "messages": updated_messages,
        }

    # Fetch trials from Elasticsearch by ID - direct sync call, no need for async/threading
    results = []
    for trial_id in trial_ids:
        try:
            # Get document by ID from Elasticsearch
            doc = es_searcher.es.client.get(index=es_searcher.index_name, id=trial_id)
            source = doc.get("_source", {})
            results.append(
                {
                    "nct_id": trial_id,
                    "title": source.get("brief_title")
                    or source.get("official_title", "N/A"),
                    "eligibility_criteria": source.get("eligibility_criteria", ""),
                    "brief_summary": source.get("brief_summary", ""),
                    "detailed_description": source.get("detailed_description", ""),
                    "locations": source.get("locations", "N/A"),
                    "phase": source.get("phase", "N/A"),
                    "status": source.get("overall_status", "N/A"),
                }
            )
        except Exception as e:
            results.append(
                {
                    "nct_id": trial_id,
                    "error": f"Trial {trial_id} not found: {str(e)}",
                }
            )

    return {"trial_lookup_results": results}


def synthesize_trial_lookup_node(state: GraphState) -> GraphState:
    """Node: Synthesize trial lookup results into a natural answer"""
    trial_results = state.get("trial_lookup_results", [])
    user_input = state.get("user_input", "")
    messages = state.get("messages", [])

    if not trial_results:
        final_answer = "I couldn't find information about the requested trial(s)."
        # Update messages
        updated_messages = list(messages) if messages else []
        updated_messages.append(AIMessage(content=final_answer))
        return {
            "final_answer": final_answer,
            "messages": updated_messages,
        }

    # Format trial information for synthesis
    trials_text = ""
    for trial in trial_results:
        if "error" in trial:
            trials_text += f"\n- {trial['nct_id']}: {trial['error']}\n"
        else:
            trials_text += f"\nTrial: {trial.get('nct_id', 'N/A')}\n"
            trials_text += f"Title: {trial.get('title', 'N/A')}\n"
            trials_text += f"Phase: {trial.get('phase', 'N/A')}\n"
            trials_text += f"Status: {trial.get('status', 'N/A')}\n"
            summary = trial.get("brief_summary", "")
            if summary:
                trials_text += f"Summary: {summary[:300]}...\n"
            eligibility = trial.get("eligibility_criteria", "")
            if eligibility:
                trials_text += f"Eligibility: {eligibility[:200]}...\n"

    # Get conversation context
    context, context_instruction = get_conversation_context(state, max_messages=3)
    conversation_context_section = (
        f"\nPREVIOUS CONVERSATION:\n{context}\n" if context else ""
    )

    # Create LLM to synthesize the answer
    synthesis_llm = ChatOpenAI(
        model=settings.llm_model, temperature=settings.temperature
    )

    prompt = trial_lookup_synthesis_prompt.format(
        user_input=user_input,
        trial_information=trials_text,
        conversation_context=conversation_context_section,
        context_instruction=context_instruction,
    )

    config = RunnableConfig(
        metadata={
            "node": "synthesize_trial_lookup",
            "operation": "trial_info_synthesis",
        },
        tags=["trial-lookup", "answer-generation"],
        run_name="synthesize_trial_lookup",
    )
    response = synthesis_llm.invoke([HumanMessage(content=prompt)], config=config)
    final_answer = response.content.strip()

    # Update messages with final answer
    updated_messages = list(messages) if messages else []
    updated_messages.append(AIMessage(content=final_answer))

    return {
        "final_answer": final_answer,
        "messages": updated_messages,
        "lookup_synthesis_prompt": prompt,
    }


def create_workflow():
    """Create and compile the LangGraph workflow.

    Returns:
        Compiled LangGraph application
    """
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("reception", reception_node)
    workflow.add_node("search", search_clinical_trials_node)
    workflow.add_node("rerank", rerank_with_llm_node)
    workflow.add_node("synthesize", synthesize_answer_node)
    workflow.add_node("lookup_trials", lookup_trials_by_id_node)
    workflow.add_node("synthesize_lookup", synthesize_trial_lookup_node)

    # Define edges with conditional routing
    workflow.set_entry_point("reception")

    # Reception routes to: lookup, search, or chitchat
    workflow.add_conditional_edges(
        "reception",
        route_query,
        {
            "lookup_trials": "lookup_trials",
            "search": "search",
            "chitchat_response": END,  # chitchat_response is handled in state
        },
    )

    # Trial lookup flow
    workflow.add_edge("lookup_trials", "synthesize_lookup")
    workflow.add_edge("synthesize_lookup", END)

    # Patient matching flow
    workflow.add_edge("search", "rerank")
    workflow.add_edge("rerank", "synthesize")
    workflow.add_edge("synthesize", END)

    # Compile with memory
    checkpointer = get_checkpointer()
    app = workflow.compile(checkpointer=checkpointer)

    return app
