"""Chainlit UI for clinical trial matching assistant."""

import chainlit as cl
from langchain_core.messages import HumanMessage

from src.core.graph import create_workflow
from src.core.state import GraphState


# Initialize the workflow
workflow = create_workflow()


@cl.on_chat_start
async def on_chat_start():
    """Initialize chat session."""
    # Store thread_id in user session
    cl.user_session.set("thread_id", cl.user_session.get("id"))
    
    # Send welcome message
    await cl.Message(
        content="👋 Welcome to the Clinical Trial Matching Assistant! I can help you:\n\n"
        "- Find clinical trials based on patient conditions\n"
        "- Summarize specific trials\n"
        "- Check eligibility for trials\n"
        "- Explain medical terms and criteria\n"
        "- Compare different trials\n\n"
        "How can I assist you today?"
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages."""
    try:
        # Get thread_id from session
        thread_id = cl.user_session.get("thread_id")
        
        # Create initial state
        initial_state: GraphState = {
            "user_input": message.content,
            "messages": [HumanMessage(content=message.content)],
            "intent_type": "",
            "patient_info": "",
            "trial_ids": [],
            "clarification_reason": "",
            "trial_data": [],
            "trial_search_query": "",
            "chitchat_response": "",
            "search_results": [],
            "reranked_results": [],
            "final_answer": "",
            "top_k": 10,
        }
        
        # Show a loading message
        msg = cl.Message(content="🔍 Processing your request...")
        await msg.send()
        
        # Run the workflow
        config = {"configurable": {"thread_id": thread_id}}
        final_state = await workflow.ainvoke(initial_state, config)
        
        # Extract the final answer from the state
        final_answer = final_state.get("final_answer", "")
        
        # If there's a chitchat response, use that instead
        if not final_answer and final_state.get("chitchat_response"):
            final_answer = final_state["chitchat_response"]
        
        # If still no answer, provide a fallback
        if not final_answer:
            final_answer = "I apologize, but I couldn't generate a response. Please try rephrasing your question."
        
        # Update the message with the final answer
        msg.content = final_answer
        await msg.update()
        
    except Exception as e:
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        await cl.Message(content=error_msg).send()


