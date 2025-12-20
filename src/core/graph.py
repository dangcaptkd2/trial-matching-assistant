"""LangGraph workflow definition and nodes for clinical trial matching."""

import asyncio

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from src.config.settings import settings
from src.core.memory import get_checkpointer
from src.core.schemas import (
    IntentClassification,
    RerankScore,
)
from src.core.state import GraphState
from src.prompts import prompts
from src.services.search import ElasticsearchTrialSearcher

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


def get_conversation_context(state: GraphState, max_messages: int = 3) -> tuple[str, str]:
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


def intent_classification_node(state: GraphState) -> GraphState:
    """Node: Intent Classification - Let LLM decide intent and what info is provided"""
    user_input = state.get("user_input", "")
    messages = state.get("messages", [])

    if not user_input:
        return {
            "intent_type": "NEEDS_CLARIFICATION",
            "patient_info": None,
            "trial_ids": None,
            "clarification_reason": "empty input",
        }

    # Get conversation context using hybrid approach
    context, context_instruction = get_conversation_context(state, max_messages=3)

    # Create LLM instance with structured output
    intent_llm = ChatOpenAI(model=settings.llm_model, temperature=settings.temperature, base_url=settings.llm_model_url)
    structured_llm = intent_llm.with_structured_output(IntentClassification)

    # Format prompt with context
    conversation_context_section = f"\nPREVIOUS CONVERSATION:\n{context}\n" if context else ""

    prompt = prompts.intent_classification_prompt.format(
        user_input=user_input,
        conversation_context=conversation_context_section,
        context_instruction=context_instruction,
    )

    # Use config to identify this node in LangSmith
    config = RunnableConfig(
        metadata={"node": "intent_classification"},
        tags=["intent_classification"],
        run_name="intent_classification",
    )

    # Get structured output
    result = structured_llm.invoke([HumanMessage(content=prompt)], config=config)

    # Update messages with user input
    updated_messages = list(messages) if messages else []
    updated_messages.append(HumanMessage(content=user_input))

    # Extract data from structured output
    patient_info = result.patient_info
    trial_ids = result.trial_ids

    # Normalize: convert null to None, ensure trial_ids is list or None
    if patient_info is None or patient_info == "":
        patient_info = None
    if trial_ids is None:
        trial_ids = None
    elif isinstance(trial_ids, list):
        # Normalize trial IDs to uppercase
        trial_ids = [tid.upper() if isinstance(tid, str) else tid for tid in trial_ids if tid]
        if not trial_ids:
            trial_ids = None

    # Return LLM's extracted data
    return {
        "messages": updated_messages,
        "intent_type": result.intent_type.value,
        "patient_info": patient_info,
        "trial_ids": trial_ids,
        "clarification_reason": result.clarification_reason,
    }


def reception_node(state: GraphState) -> GraphState:
    """Node: Reception - Handle greetings and off-topic queries"""
    user_input = state.get("user_input", "")
    intent_type = state.get("intent_type", "")
    messages = state.get("messages", [])

    # Get conversation context
    context, context_instruction = get_conversation_context(state, max_messages=3)

    # Create LLM instance for reception
    reception_llm = ChatOpenAI(
        model=settings.llm_model, temperature=settings.temperature, base_url=settings.llm_model_url
    )

    # Format prompt with context
    conversation_context_section = f"\nPREVIOUS CONVERSATION:\n{context}\n" if context else ""

    prompt = prompts.reception_prompt.format(
        user_input=user_input,
        intent_type=intent_type,
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
    chitchat_response = response.content.strip()

    # Update messages with assistant response
    updated_messages = list(messages) if messages else []
    updated_messages.append(AIMessage(content=chitchat_response))

    return {
        "messages": updated_messages,
        "chitchat_response": chitchat_response,
    }


def translate_terms_node(state: GraphState) -> GraphState:
    """Node: Translate medical terms - Explain medical jargon in simple language"""
    user_input = state.get("user_input", "")
    messages = state.get("messages", [])

    # Get conversation context
    context, context_instruction = get_conversation_context(state, max_messages=3)

    # Create LLM instance
    # Note: translate_terms returns natural language, keeping as-is for now
    translate_llm = ChatOpenAI(
        model=settings.llm_model, temperature=settings.temperature, base_url=settings.llm_model_url
    )

    # Format prompt with context
    conversation_context_section = f"\nPREVIOUS CONVERSATION:\n{context}\n" if context else ""

    prompt = prompts.translate_terms_prompt.format(
        user_input=user_input,
        conversation_context=conversation_context_section,
        context_instruction=context_instruction,
    )

    # Use config to identify this node in LangSmith
    config = RunnableConfig(
        metadata={"node": "translate_terms"},
        tags=["translate_terms", "term_explanation"],
        run_name="translate_terms",
    )

    response = translate_llm.invoke([HumanMessage(content=prompt)], config=config)
    final_answer = response.content.strip()

    # Update messages with assistant response
    updated_messages = list(messages) if messages else []
    updated_messages.append(AIMessage(content=final_answer))

    return {
        "messages": updated_messages,
        "final_answer": final_answer,
    }


def route_from_intent(state: GraphState) -> str:
    """Conditional edge function: route based on intent type"""
    intent_type = state.get("intent_type", "NEEDS_CLARIFICATION")

    if intent_type == "GREETING" or intent_type == "OFF_TOPIC":
        return "reception"
    elif intent_type == "TRANSLATE_TERMS":
        # Translate medical terms - no data needed
        return "translate_terms"
    elif intent_type == "FIND_TRIALS":
        # Always search - intent type means they have patient info
        return "search"
    elif intent_type == "SUMMARIZE_TRIAL":
        # Fetch trial data first, then summarize
        return "fetch_trial_data"
    elif intent_type == "CHECK_ELIGIBILITY":
        # Fetch trial data first, then check eligibility
        return "fetch_trial_data"
    elif intent_type == "EXPLAIN_CRITERIA":
        # Fetch trial data first, then explain criteria
        return "fetch_trial_data"
    elif intent_type == "COMPARE_TRIALS":
        # Fetch trial data first, then compare
        return "fetch_trial_data"
    else:  # NEEDS_CLARIFICATION
        return "clarify"


def route_after_fetch(state: GraphState) -> str:
    """Route after fetching trial data based on intent type"""
    intent_type = state.get("intent_type", "")

    if intent_type == "SUMMARIZE_TRIAL":
        return "summarize_trial"
    elif intent_type == "CHECK_ELIGIBILITY":
        return "check_eligibility"
    elif intent_type == "EXPLAIN_CRITERIA":
        return "explain_criteria"
    elif intent_type == "COMPARE_TRIALS":
        return "compare_trials"
    else:
        # Should never happen - fetch_trial_data only runs for SUMMARIZE/CHECK_ELIGIBILITY/EXPLAIN_CRITERIA/COMPARE_TRIALS
        return "summarize_trial"  # Safe fallback


async def search_clinical_trials_node(state: GraphState) -> GraphState:
    """Node: Search clinical trials using Elasticsearch"""
    patient_info = state.get("patient_info", "")
    top_k = state.get("top_k", 10)

    if not patient_info:
        return {"search_results": []}

    # Directly call the async function - no need for event loop juggling!
    search_results = await es_searcher.get_trials_by_text(
        patient_profile=patient_info,
        top_k=top_k,
        return_fields=[
            "eligibility_criteria",
            "official_title",
        ],
    )

    # Format results with eligibility criteria
    formatted_results = []
    for result in search_results:
        nct_id = result.get("nct_id") or result.get("id") or result.get("id", "N/A")
        title = result.get("official_title")
        eligibility = result.get("eligibility_criteria", "")
        distance = result.get("distance", 0.0)

        formatted_results.append(
            {
                "nct_id": nct_id,
                "title": title,
                "eligibility_criteria": eligibility,
                "es_score": distance,
            }
        )
    return {"search_results": formatted_results}


async def rerank_with_llm_node(state: GraphState) -> GraphState:
    """Node: Rerank search results using LLM as cross-encoder"""
    search_results = state.get("search_results", [])
    patient_info = state.get("patient_info", "")

    if not search_results:
        return {"reranked_results": []}

    # Create LLM instance with structured output for cross-encoding
    cross_encoder_llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.temperature,
        base_url=settings.llm_model_url,
        max_tokens=4096,
    )
    structured_llm = cross_encoder_llm.with_structured_output(RerankScore)

    async def score_trial(trial: dict) -> dict:
        """Use LLM to score a single trial against patient profile"""
        eligibility = trial.get("eligibility_criteria", "")
        if not eligibility:
            return {
                **trial,
                "llm_score": 0.0,
                "match_reasoning": "No eligibility criteria available",
            }

        # Truncate eligibility criteria to prevent token limit issues
        # Keep first 800 characters (roughly 600-800 tokens) to leave room for
        # patient profile, prompt template, and response
        if len(eligibility) > 800:
            eligibility = eligibility[:800] + "\n\n[... eligibility criteria truncated for length ...]"

        prompt = prompts.rerank_prompt.format(patient_profile=patient_info, eligibility=eligibility)

        config = RunnableConfig(
            metadata={"node": "rerank_with_llm", "operation": "trial_scoring"},
            tags=["rerank", "cross-encoder", "trial-matching"],
            run_name="rerank_trial_scoring",
        )

        # Get structured output
        result = await structured_llm.ainvoke([HumanMessage(content=prompt)], config=config)

        return {
            **trial,
            "llm_score": result.score,
            "match_reasoning": result.reasoning,
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
    patient_info = state.get("patient_info", "")
    user_input = state.get("user_input", "")
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
    conversation_context_section = f"\nPREVIOUS CONVERSATION:\n{context}\n" if context else ""

    # Create LLM instance for synthesis
    # Note: synthesis returns natural language, keeping as-is for now
    synthesis_llm = ChatOpenAI(
        model=settings.llm_model, temperature=settings.temperature, base_url=settings.llm_model_url
    )

    prompt = prompts.synthesis_prompt.format(
        user_input=user_input,
        patient_profile=patient_info,
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
    """Node: Generate clarification response when user needs to provide more information"""
    user_input = state.get("user_input", "")
    messages = state.get("messages", [])
    clarification_reason = state.get("clarification_reason", "")
    patient_info = state.get("patient_info")
    trial_ids = state.get("trial_ids", [])

    # Check if info is available
    has_patient_info = patient_info is not None and patient_info != ""
    has_trial_id = trial_ids is not None and len(trial_ids) > 0

    # Get conversation context
    context, context_instruction = get_conversation_context(state, max_messages=3)
    conversation_context_section = f"\nPREVIOUS CONVERSATION:\n{context}\n" if context else ""

    # Determine what needs clarification based on flags and reason
    clarification_context = clarification_reason
    search_results_section = ""
    clarification_instructions = ""

    # Check clarification reason to understand what user wanted
    reason_lower = clarification_reason.lower()
    wants_find_trials = "find" in reason_lower or "search" in reason_lower
    wants_summarize = "summary" in reason_lower or "summarize" in reason_lower
    wants_check_eligibility = "eligible" in reason_lower or "eligibility" in reason_lower or "qualify" in reason_lower
    wants_explain_criteria = (
        "explain" in reason_lower
        or "criteria" in reason_lower
        or "eligibility rules" in reason_lower
        or "inclusion" in reason_lower
        or "exclusion" in reason_lower
    )
    wants_compare = "compare" in reason_lower or "comparison" in reason_lower

    if wants_find_trials and not has_patient_info:
        # User wants to find trials but no patient information provided
        clarification_instructions = "Ask the user to provide patient information such as age, gender, medical condition, diagnosis, symptoms, or current treatment status. Keep it conversational and friendly."
    elif wants_summarize and not has_trial_id:
        # User wants to summarize but no trial ID provided
        if has_patient_info:
            # User provided patient info but no trial ID - search and show options
            search_query = user_input
            clarification_search_results = await es_searcher.get_trials_by_text(
                search_query,
                top_k=5,
                return_fields=["nct_id", "official_title", "brief_summary"],
            )
            if clarification_search_results:
                search_results_section = "SEARCH RESULTS:\n"
                for i, trial in enumerate(clarification_search_results, 1):
                    nct_id = trial.get("nct_id", "N/A")
                    title = trial.get("official_title", "N/A")
                    brief_summary = trial.get("brief_summary", "N/A")
                    search_results_section += f"{i}. {nct_id}: {title}\n   {brief_summary[:200]}...\n\n"
                clarification_instructions = "Present the search results in a clear, numbered list and ask the user to provide the specific trial ID (NCT number) they want to summarize."
                clarification_context = f"Found {len(clarification_search_results)} matching trials"
            else:
                clarification_instructions = (
                    "Briefly explain that no trials were found and ask for a specific trial ID (NCT number)."
                )
        else:
            # No patient info and no trial ID - just ask for trial ID
            clarification_instructions = (
                "Ask the user to provide a trial ID (format: NCT12345678) to summarize. Keep it short and friendly."
            )
    elif wants_check_eligibility:
        if not has_patient_info and not has_trial_id:
            clarification_instructions = "Ask for BOTH patient information (age, condition) AND the specific trial ID (NCT number) to check eligibility."
        elif not has_patient_info:
            clarification_instructions = (
                "Ask for patient information (age, condition, diagnosis) to check eligibility for the specific trial."
            )
        elif not has_trial_id:
            clarification_instructions = "Ask for the specific trial ID (NCT number) to check eligibility against."
    elif wants_explain_criteria and not has_trial_id:
        # User wants to explain criteria but no trial ID provided - just ask for trial ID
        clarification_instructions = "Ask the user to provide a trial ID (format: NCT12345678) to explain the eligibility criteria. Keep it short and friendly."
    elif wants_compare:
        # User wants to compare trials
        if not has_trial_id:
            clarification_instructions = "Ask the user to provide at least 2 trial IDs (format: NCT12345678) to compare. Keep it short and friendly."
        elif len(trial_ids) < 2:
            clarification_instructions = f"Ask the user to provide at least one more trial ID (currently have {len(trial_ids)}). Need at least 2 trials to compare. Keep it short and friendly."
    else:
        # Generic clarification - truly ambiguous or empty input
        clarification_instructions = "Ask the user to provide more information to proceed with their request."

    # Create LLM instance for clarification
    clarify_llm = ChatOpenAI(
        model=settings.llm_model, temperature=settings.temperature, base_url=settings.llm_model_url
    )

    prompt = prompts.clarification_prompt.format(
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


def fetch_trial_data_node(state: GraphState) -> GraphState:
    """Node: Fetch trial documents from Elasticsearch by IDs"""
    trial_ids = state.get("trial_ids", [])

    if not trial_ids:
        return {
            "trial_data": [],
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
                    "title": source.get("brief_title") or source.get("official_title", "N/A"),
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

    return {"trial_data": trial_results}


def summarize_trial_node(state: GraphState) -> GraphState:
    """Node: Summarize specific clinical trial(s) by their IDs"""
    trial_data = state.get("trial_data", [])
    messages = state.get("messages", [])
    user_input = state.get("user_input", "")

    if not trial_data or all("error" in trial for trial in trial_data):
        final_answer = "I couldn't find information about the requested trial(s)."
        updated_messages = list(messages) if messages else []
        updated_messages.append(AIMessage(content=final_answer))
        return {
            "final_answer": final_answer,
            "messages": updated_messages,
        }

    # Format trial information for summarization
    trials_text = ""
    for trial in trial_data:
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
                trials_text += f"Description: {trial.get('detailed_description')[:500]}...\n"
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
    conversation_context_section = f"\nPREVIOUS CONVERSATION:\n{context}\n" if context else ""

    # Create LLM to summarize the trial(s)
    synthesis_llm = ChatOpenAI(
        model=settings.llm_model, temperature=settings.temperature, base_url=settings.llm_model_url
    )

    prompt = prompts.summarize_trial_prompt.format(
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


async def check_eligibility_node(state: GraphState) -> GraphState:
    """Node: Check if a patient is eligible for a specific trial"""
    trial_data = state.get("trial_data", [])
    patient_info = state.get("patient_info", "")
    user_input = state.get("user_input", "")
    messages = state.get("messages", [])

    if not trial_data:
        final_answer = (
            "I couldn't find trial information to check eligibility. Please provide a trial ID (e.g., NCT12345678)."
        )
        updated_messages = list(messages) if messages else []
        updated_messages.append(AIMessage(content=final_answer))
        return {
            "final_answer": final_answer,
            "messages": updated_messages,
        }

    # Check eligibility for the first trial (focus on primary request)
    trial = trial_data[0]

    if "error" in trial:
        final_answer = (
            f"I couldn't find details for trial {trial.get('nct_id', 'N/A')}. {trial.get('error', 'Trial not found')}"
        )
        updated_messages = list(messages) if messages else []
        updated_messages.append(AIMessage(content=final_answer))
        return {
            "final_answer": final_answer,
            "messages": updated_messages,
        }

    target_trial_id = trial.get("nct_id", "N/A")
    eligibility_criteria = trial.get("eligibility_criteria", "Not available")

    # Get conversation context
    context, context_instruction = get_conversation_context(state, max_messages=3)
    conversation_context_section = f"\nPREVIOUS CONVERSATION:\n{context}\n" if context else ""

    # Create LLM instance
    check_llm = ChatOpenAI(model=settings.llm_model, temperature=settings.temperature, base_url=settings.llm_model_url)

    prompt = prompts.check_eligibility_prompt.format(
        user_input=user_input,
        conversation_context=conversation_context_section,
        patient_profile=patient_info,
        trial_id=target_trial_id,
        eligibility_criteria=eligibility_criteria,
        context_instruction=context_instruction,
    )

    config = RunnableConfig(
        metadata={"node": "check_eligibility", "operation": "eligibility_check"},
        tags=["eligibility", "analysis"],
        run_name="check_eligibility",
    )

    response = check_llm.invoke([HumanMessage(content=prompt)], config=config)
    final_answer = response.content.strip()

    # Update messages
    updated_messages = list(messages) if messages else []
    updated_messages.append(AIMessage(content=final_answer))

    return {
        "final_answer": final_answer,
        "messages": updated_messages,
    }


def explain_criteria_node(state: GraphState) -> GraphState:
    """Node: Explain eligibility criteria in simple, everyday language"""
    trial_data = state.get("trial_data", [])
    messages = state.get("messages", [])
    user_input = state.get("user_input", "")

    if not trial_data or all("error" in trial for trial in trial_data):
        final_answer = "I couldn't find information about the requested trial(s). Please provide a valid trial ID (e.g., NCT12345678)."
        updated_messages = list(messages) if messages else []
        updated_messages.append(AIMessage(content=final_answer))
        return {
            "final_answer": final_answer,
            "messages": updated_messages,
        }

    # Get conversation context
    context, context_instruction = get_conversation_context(state, max_messages=3)
    conversation_context_section = f"\nPREVIOUS CONVERSATION:\n{context}\n" if context else ""

    # Create LLM instance
    # Note: explain_criteria returns natural language. Could use CriteriaExplanation schema in future
    explain_llm = ChatOpenAI(
        model=settings.llm_model, temperature=settings.temperature, base_url=settings.llm_model_url
    )

    # Process each trial separately
    explanations = []
    for trial in trial_data:
        if "error" in trial:
            explanations.append(f"\n**Trial {trial.get('nct_id', 'N/A')}:** {trial.get('error', 'Trial not found')}\n")
            continue

        trial_id = trial.get("nct_id", "N/A")
        trial_title = trial.get("title", "N/A")
        eligibility_criteria = trial.get("eligibility_criteria", "Not available")

        prompt = prompts.explain_criteria_prompt.format(
            conversation_context=conversation_context_section,
            trial_id=trial_id,
            trial_title=trial_title,
            eligibility_criteria=eligibility_criteria,
            context_instruction=context_instruction,
            user_input=user_input,
        )

        config = RunnableConfig(
            metadata={"node": "explain_criteria", "operation": "criteria_explanation"},
            tags=["criteria-explanation", "answer-generation"],
            run_name="explain_criteria",
        )

        response = explain_llm.invoke([HumanMessage(content=prompt)], config=config)
        explanation = response.content.strip()

        # Add trial header if multiple trials
        if len(trial_data) > 1:
            explanations.append(f"\n## {trial_title} ({trial_id})\n\n{explanation}\n")
        else:
            explanations.append(explanation)

    # Combine explanations
    final_answer = "\n".join(explanations)

    # Update messages with final answer
    updated_messages = list(messages) if messages else []
    updated_messages.append(AIMessage(content=final_answer))

    return {
        "final_answer": final_answer,
        "messages": updated_messages,
    }


def compare_trials_node(state: GraphState) -> GraphState:
    """Node: Compare 2 or more clinical trials side by side"""
    trial_data = state.get("trial_data", [])
    messages = state.get("messages", [])
    user_input = state.get("user_input", "")

    if not trial_data or all("error" in trial for trial in trial_data):
        final_answer = "I couldn't find information about the requested trials for comparison. Please provide valid trial IDs (e.g., NCT12345678)."
        updated_messages = list(messages) if messages else []
        updated_messages.append(AIMessage(content=final_answer))
        return {
            "final_answer": final_answer,
            "messages": updated_messages,
        }

    # Check if we have at least 2 valid trials
    valid_trials = [t for t in trial_data if "error" not in t]
    if len(valid_trials) < 2:
        final_answer = "I need at least 2 valid trials to compare. Please provide 2 or more trial IDs."
        updated_messages = list(messages) if messages else []
        updated_messages.append(AIMessage(content=final_answer))
        return {
            "final_answer": final_answer,
            "messages": updated_messages,
        }

    # Format trial information for comparison
    trials_text = ""
    for trial in trial_data:
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
                trials_text += f"Description: {trial.get('detailed_description')[:500]}...\n"
            if trial.get("eligibility_criteria"):
                trials_text += f"Eligibility: {trial.get('eligibility_criteria')}\n"
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
    conversation_context_section = f"\nPREVIOUS CONVERSATION:\n{context}\n" if context else ""

    # Create LLM to compare the trials
    # Note: compare_trials returns natural language. Could use TrialComparison schema in future
    compare_llm = ChatOpenAI(
        model=settings.llm_model, temperature=settings.temperature, base_url=settings.llm_model_url
    )

    prompt = prompts.compare_trials_prompt.format(
        user_input=user_input,
        trial_information=trials_text,
        conversation_context=conversation_context_section,
        context_instruction=context_instruction,
    )

    config = RunnableConfig(
        metadata={"node": "compare_trials", "operation": "trial_comparison"},
        tags=["trial-comparison", "answer-generation"],
        run_name="compare_trials",
    )
    response = compare_llm.invoke([HumanMessage(content=prompt)], config=config)
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
    workflow.add_node("intent_classification", intent_classification_node)
    workflow.add_node("reception", reception_node)
    workflow.add_node("translate_terms", translate_terms_node)
    workflow.add_node("clarify", clarify_node)
    workflow.add_node("fetch_trial_data", fetch_trial_data_node)
    workflow.add_node("check_eligibility", check_eligibility_node)
    workflow.add_node("explain_criteria", explain_criteria_node)
    workflow.add_node("compare_trials", compare_trials_node)
    workflow.add_node("search", search_clinical_trials_node)
    workflow.add_node("rerank", rerank_with_llm_node)
    workflow.add_node("synthesize", synthesize_answer_node)
    workflow.add_node("summarize_trial", summarize_trial_node)

    # Define edges with conditional routing
    workflow.set_entry_point("intent_classification")

    # Intent classification routes to appropriate nodes
    workflow.add_conditional_edges(
        "intent_classification",
        route_from_intent,
        {
            "reception": "reception",
            "translate_terms": "translate_terms",
            "search": "search",
            "fetch_trial_data": "fetch_trial_data",
            "clarify": "clarify",
        },
    )

    # Reception (greeting/off-topic) flow
    workflow.add_edge("reception", END)

    # Translate terms flow
    workflow.add_edge("translate_terms", END)

    # Clarification flow
    workflow.add_edge("clarify", END)

    # Fetch trial data routes to summarize, check eligibility, explain criteria, or compare (always has required info)
    workflow.add_conditional_edges(
        "fetch_trial_data",
        route_after_fetch,
        {
            "summarize_trial": "summarize_trial",
            "check_eligibility": "check_eligibility",
            "explain_criteria": "explain_criteria",
            "compare_trials": "compare_trials",
        },
    )

    # Trial summarization flow
    workflow.add_edge("summarize_trial", END)

    # Eligibility check flow
    workflow.add_edge("check_eligibility", END)

    # Explain criteria flow
    workflow.add_edge("explain_criteria", END)

    # Compare trials flow
    workflow.add_edge("compare_trials", END)

    # Find trials flow (patient matching)
    workflow.add_edge("search", "rerank")
    workflow.add_edge("rerank", "synthesize")
    workflow.add_edge("synthesize", END)

    # Compile with memory
    checkpointer = get_checkpointer()
    app = workflow.compile(checkpointer=checkpointer)

    return app
