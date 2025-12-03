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
    summarize_trial_prompt,
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
            "query_type": "CHITCHAT",
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

    result = json.loads(content)

    # Update messages with user input
    updated_messages = list(messages) if messages else []
    updated_messages.append(HumanMessage(content=user_input))

    # Add assistant response if chitchat
    if result.get("chitchat_response"):
        updated_messages.append(AIMessage(content=result.get("chitchat_response", "")))

    return {
        "messages": updated_messages,
        "query_type": result.get("query_type", "CHITCHAT"),
        "clarification_type": result.get("clarification_type", ""),
        "clarification_context": result.get("clarification_context", ""),
        "trial_search_query": result.get("trial_search_query", ""),
        "trial_ids": result.get("trial_ids", []),
        "chitchat_response": result.get("chitchat_response", ""),
        "search_query": result.get("search_query", ""),
        "patient_profile": result.get("patient_profile", ""),
    }


def route_query(state: GraphState) -> str:
    """Conditional edge function: route based on query type"""
    query_type = state.get("query_type", "CHITCHAT")

    if query_type == "CLARIFY":
        return "clarify"
    elif query_type == "SUMMARIZE_TRIAL":
        return "summarize_trial"
    elif query_type == "FIND_TRIALS":
        return "search"
    else:  # CHITCHAT
        return "chitchat_response"


async def search_clinical_trials_node(state: GraphState) -> GraphState:
    """Node: Search clinical trials using Elasticsearch"""
    query = state.get("search_query", "")
    patient_profile = state.get("patient_profile", "")
    top_k = state.get("top_k", 10)

    # Use patient_profile if available, otherwise use search_query
    search_text = patient_profile if patient_profile else query

    # Directly call the async function - no need for event loop juggling!
    search_results = await es_searcher.get_trials_by_text(
        search_text,
        top_k=top_k,
        return_fields=[
            "eligibility_criteria",
            "official_title",
        ],
    )

    # Format results with eligibility criteria
    formatted_results = []
    for result in search_results:
        source = result.get("source", {})
        nct_id = source.get("nct_id") or source.get("id") or result.get("id", "N/A")
        title = source.get("official_title")
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


async def clarify_node(state: GraphState) -> GraphState:
    """Node: Generate clarification response using LLM when user needs to provide more information"""
    clarification_type = state.get("clarification_type", "")
    clarification_context = state.get("clarification_context", "")
    trial_search_query = state.get("trial_search_query", "")
    user_input = state.get("user_input", "")
    messages = state.get("messages", [])

    # Get conversation context
    context, context_instruction = get_conversation_context(state, max_messages=3)
    conversation_context_section = (
        f"\nPREVIOUS CONVERSATION:\n{context}\n" if context else ""
    )

    # Perform search if needed
    clarification_search_results = []
    if clarification_type == "trial_id" and trial_search_query:
        try:
            clarification_search_results = await es_searcher.get_trials_by_text(
                trial_search_query,
                top_k=5,
                return_fields=["nct_id", "official_title", "brief_summary"],
            )
            if clarification_search_results:
                clarification_context += f" I searched and found {len(clarification_search_results)} matching trials."
            else:
                clarification_context += f" I searched but found no matching trials for '{trial_search_query}'."
        except Exception as e:
            clarification_context += (
                f" I encountered an error while searching: {str(e)}"
            )

    # Format search results and instructions based on what information we have
    search_results_section = ""
    clarification_instructions = ""

    if clarification_type == "trial_id":
        if clarification_search_results:
            # Case 2a: Search performed and found results
            search_results_section = "SEARCH RESULTS:\n"
            for i, trial in enumerate(clarification_search_results, 1):
                nct_id = trial.get("nct_id", "N/A")
                title = trial.get("official_title", "N/A")
                brief_summary = trial.get("brief_summary", "N/A")
                search_results_section += f"{i}. {nct_id}: {title}\n{brief_summary}\n"
            clarification_instructions = "Present the search results in a clear, numbered list and ask the user to provide the specific trial ID (NCT number) they want to summarize."
        elif trial_search_query:
            # Case 2b: Search performed but no results found
            clarification_instructions = f"Briefly explain that no trials were found for '{trial_search_query}' and ask for a specific trial ID (NCT number)."
        else:
            # Case 1: No search performed - user gave no information
            clarification_instructions = "Briefly ask the user to provide a trial ID (format: NCT12345678) to summarize. Keep it short and friendly."
    elif clarification_type == "patient_profile":
        clarification_instructions = "Asks the user to provide patient information such as age, gender, medical condition, diagnosis, or current treatment status"
    else:
        clarification_instructions = (
            "Asks the user to provide the missing information needed to proceed"
        )

    # Create LLM instance for clarification
    clarify_llm = ChatOpenAI(model=settings.llm_model, temperature=settings.temperature)

    from src.prompts.prompts import clarification_prompt

    prompt = clarification_prompt.format(
        user_input=user_input,
        clarification_context=clarification_context,
        clarification_instructions=clarification_instructions,
        search_results_section=search_results_section,
        conversation_context=conversation_context_section,
        context_instruction=context_instruction,
    )

    config = RunnableConfig(
        metadata={
            "node": "clarify",
            "operation": "clarification",
        },
        tags=["clarification", "user-interaction"],
        run_name="clarify",
    )
    response = clarify_llm.invoke([HumanMessage(content=prompt)], config=config)
    clarification_response = response.content.strip()

    # Update messages with clarification response
    updated_messages = list(messages) if messages else []
    updated_messages.append(AIMessage(content=clarification_response))

    return {
        "chitchat_response": clarification_response,
        "messages": updated_messages,
    }


def summarize_trial_node(state: GraphState) -> GraphState:
    """Node: Summarize specific clinical trial(s) by their IDs"""
    trial_ids = state.get("trial_ids", [])
    messages = state.get("messages", [])
    user_input = state.get("user_input", "")

    if not trial_ids:
        final_answer = "I couldn't find any trial IDs in your query. Please provide trial IDs in the format NCT12345678."
        updated_messages = list(messages) if messages else []
        updated_messages.append(AIMessage(content=final_answer))
        return {
            "final_answer": final_answer,
            "messages": updated_messages,
        }

    # Fetch trials from Elasticsearch by ID
    trial_results = []
    for trial_id in trial_ids:
        try:
            doc = es_searcher.es.client.get(index=es_searcher.index_name, id=trial_id)
            source = doc.get("_source", {})
            trial_results.append(
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
                    "start_date": source.get("start_date", "N/A"),
                    "completion_date": source.get("completion_date", "N/A"),
                    "primary_outcome": source.get("primary_outcome_measure", "N/A"),
                }
            )
        except Exception as e:
            trial_results.append(
                {
                    "nct_id": trial_id,
                    "error": f"Trial {trial_id} not found: {str(e)}",
                }
            )

    if not trial_results or all("error" in trial for trial in trial_results):
        final_answer = "I couldn't find information about the requested trial(s)."
        updated_messages = list(messages) if messages else []
        updated_messages.append(AIMessage(content=final_answer))
        return {
            "final_answer": final_answer,
            "messages": updated_messages,
        }

    # Format trial information for summarization
    trials_text = ""
    for trial in trial_results:
        if "error" in trial:
            trials_text += f"\n- {trial['nct_id']}: {trial['error']}\n"
        else:
            trials_text += f"\nTrial ID: {trial.get('nct_id', 'N/A')}\n"
            trials_text += f"Title: {trial.get('title', 'N/A')}\n"
            trials_text += f"Phase: {trial.get('phase', 'N/A')}\n"
            trials_text += f"Status: {trial.get('status', 'N/A')}\n"
            if trial.get("brief_summary"):
                trials_text += f"Summary: {trial.get('brief_summary')}\n"
            if trial.get("detailed_description"):
                trials_text += (
                    f"Description: {trial.get('detailed_description')[:500]}...\n"
                )
            if trial.get("eligibility_criteria"):
                trials_text += f"Eligibility: {trial.get('eligibility_criteria')}...\n"
            if trial.get("primary_outcome") and trial.get("primary_outcome") != "N/A":
                trials_text += f"Primary Outcome: {trial.get('primary_outcome')}\n"
            if trial.get("start_date") and trial.get("start_date") != "N/A":
                trials_text += f"Start Date: {trial.get('start_date')}\n"
            if trial.get("completion_date") and trial.get("completion_date") != "N/A":
                trials_text += f"Completion Date: {trial.get('completion_date')}\n"
            if trial.get("locations") and trial.get("locations") != "N/A":
                trials_text += f"Locations: {trial.get('locations')}\n"

    # Get conversation context
    context, context_instruction = get_conversation_context(state, max_messages=3)
    conversation_context_section = (
        f"\nPREVIOUS CONVERSATION:\n{context}\n" if context else ""
    )

    # Create LLM to summarize the trial(s)
    synthesis_llm = ChatOpenAI(
        model=settings.llm_model, temperature=settings.temperature
    )

    prompt = summarize_trial_prompt.format(
        user_input=user_input,
        trial_information=trials_text,
        conversation_context=conversation_context_section,
        context_instruction=context_instruction,
    )

    config = RunnableConfig(
        metadata={
            "node": "summarize_trial",
            "operation": "trial_summarization",
        },
        tags=["trial-summary", "answer-generation"],
        run_name="summarize_trial",
    )
    response = synthesis_llm.invoke([HumanMessage(content=prompt)], config=config)
    final_answer = response.content.strip()

    # Update messages with final answer
    updated_messages = list(messages) if messages else []
    updated_messages.append(AIMessage(content=final_answer))

    return {
        "final_answer": final_answer,
        "messages": updated_messages,
    }


def create_workflow():
    """Create and compile the LangGraph workflow.

    Returns:
        Compiled LangGraph application
    """
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("reception", reception_node)
    workflow.add_node("clarify", clarify_node)
    workflow.add_node("search", search_clinical_trials_node)
    workflow.add_node("rerank", rerank_with_llm_node)
    workflow.add_node("synthesize", synthesize_answer_node)
    workflow.add_node("summarize_trial", summarize_trial_node)

    # Define edges with conditional routing
    workflow.set_entry_point("reception")

    # Reception routes to: clarify, summarize_trial, search, or chitchat
    workflow.add_conditional_edges(
        "reception",
        route_query,
        {
            "clarify": "clarify",
            "summarize_trial": "summarize_trial",
            "search": "search",
            "chitchat_response": END,  # chitchat_response is handled in state
        },
    )

    # Clarification flow
    workflow.add_edge("clarify", END)

    # Trial summarization flow
    workflow.add_edge("summarize_trial", END)

    # Find trials flow (patient matching)
    workflow.add_edge("search", "rerank")
    workflow.add_edge("rerank", "synthesize")
    workflow.add_edge("synthesize", END)

    # Compile with memory
    checkpointer = get_checkpointer()
    app = workflow.compile(checkpointer=checkpointer)

    return app
