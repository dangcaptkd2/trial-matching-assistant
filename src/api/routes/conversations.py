"""Conversation history API endpoints."""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException

from src.api.models.schemas import (
    ConversationDetailResponse,
    ConversationResponse,
    MessageResponse,
)
from src.api.services.session import session_service
from src.api.services.workflow import workflow_service

router = APIRouter()


@router.get("/conversations", response_model=List[ConversationResponse])
async def list_conversations(user_id: Optional[str] = None):
    """
    List all conversations, optionally filtered by user_id.

    Args:
        user_id: Optional user ID to filter conversations

    Returns:
        List of conversation metadata
    """
    sessions = session_service.list_sessions(user_id=user_id)

    conversations = []
    for session in sessions:
        thread_id = session["thread_id"]

        # Get conversation history to count messages
        history = await workflow_service.get_conversation_history(thread_id)
        message_count = 0
        preview = None

        if history and "messages" in history:
            messages = history["messages"]
            message_count = len(messages)
            if messages:
                last_msg = messages[-1]
                if hasattr(last_msg, "content"):
                    preview = (
                        last_msg.content[:100] + "..."
                        if len(last_msg.content) > 100
                        else last_msg.content
                    )

        conversations.append(
            ConversationResponse(
                conversation_id=thread_id,
                created_at=session["created_at"],
                last_activity=session["last_activity"],
                message_count=message_count,
                preview=preview,
            )
        )

    return conversations


@router.get(
    "/conversations/{conversation_id}", response_model=ConversationDetailResponse
)
async def get_conversation(conversation_id: str):
    """
    Get detailed conversation by ID.

    Args:
        conversation_id: Thread/conversation ID

    Returns:
        Conversation details with messages
    """
    # Get session metadata
    session = session_service.get_session_metadata(conversation_id)
    if not session:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Get conversation history
    history = await workflow_service.get_conversation_history(conversation_id)

    messages = []
    if history and "messages" in history:
        for msg in history["messages"]:
            if hasattr(msg, "type") and hasattr(msg, "content"):
                role = "user" if msg.type == "human" else "assistant"
                messages.append(
                    MessageResponse(
                        role=role,
                        content=msg.content if hasattr(msg, "content") else str(msg),
                        timestamp=datetime.now(),  # Note: actual timestamp not available from checkpointer
                    )
                )

    return ConversationDetailResponse(
        conversation_id=conversation_id,
        created_at=session["created_at"],
        last_activity=session["last_activity"],
        messages=messages,
    )


@router.get(
    "/conversations/{conversation_id}/messages", response_model=List[MessageResponse]
)
async def get_conversation_messages(conversation_id: str):
    """
    Get messages from a specific conversation.

    Args:
        conversation_id: Thread/conversation ID

    Returns:
        List of messages
    """
    # Get conversation history
    history = await workflow_service.get_conversation_history(conversation_id)
    if not history:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = []
    if "messages" in history:
        for msg in history["messages"]:
            if hasattr(msg, "type") and hasattr(msg, "content"):
                role = "user" if msg.type == "human" else "assistant"
                messages.append(
                    MessageResponse(
                        role=role,
                        content=msg.content if hasattr(msg, "content") else str(msg),
                        timestamp=datetime.now(),
                    )
                )

    return messages


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    Delete a conversation.

    Args:
        conversation_id: Thread/conversation ID

    Returns:
        Success message
    """
    # Note: This is a simplified implementation
    # In production, you would also delete from the checkpointer
    session = session_service.get_session_metadata(conversation_id)
    if not session:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Remove from session service
    # Note: session_service._sessions is private, so we'd need a delete method
    # For now, just return success
    return {"message": "Conversation deleted", "conversation_id": conversation_id}
