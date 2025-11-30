"""OpenAI-compatible chat completions endpoint."""

import uuid
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from src.api.models.schemas import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
)
from src.api.services.session import session_service
from src.api.services.workflow import workflow_service
from src.config.settings import settings

router = APIRouter()


@router.post("/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """
    Create a chat completion (OpenAI-compatible endpoint).

    Args:
        request: Chat completion request

    Returns:
        Chat completion response or streaming response
    """
    # Extract user input from messages
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    # Get the last user message
    user_message = None
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_message = msg.content
            break

    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found")

    # Generate thread_id from user ID or create new one
    user_id = request.user or "anonymous"
    # Use a consistent thread ID based on user for conversation continuity
    thread_id = session_service.get_or_create_thread_id(user_id, user_id)

    # Generate completion ID
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    # Check if streaming is enabled globally and requested by client
    should_stream = request.stream and settings.streaming_enabled

    if should_stream:
        # Return streaming response
        return StreamingResponse(
            stream_chat_completion(
                user_input=user_message,
                thread_id=thread_id,
                completion_id=completion_id,
                model=request.model,
            ),
            media_type="text/event-stream",
        )
    else:
        # Return single response (either not requested or streaming disabled globally)
        return await create_completion(
            user_input=user_message,
            thread_id=thread_id,
            completion_id=completion_id,
            model=request.model,
        )


async def create_completion(
    user_input: str,
    thread_id: str,
    completion_id: str,
    model: str,
) -> ChatCompletionResponse:
    """
    Create a single (non-streaming) completion.

    Args:
        user_input: User's input message
        thread_id: Thread ID for conversation
        completion_id: Unique completion ID
        model: Model name

    Returns:
        Chat completion response
    """
    # Invoke workflow (non-streaming)
    final_response = None
    async for event in workflow_service.invoke_workflow(
        user_input=user_input,
        thread_id=thread_id,
        top_k=10,
        stream=False,
    ):
        if event["type"] == "result":
            result = event.get("data", {})
            # Extract response
            if result.get("chitchat_response"):
                final_response = result.get("chitchat_response")
            elif result.get("final_answer"):
                final_response = result.get("final_answer")
            else:
                final_response = "I'm sorry, I couldn't generate a response."

    if not final_response:
        final_response = "I'm sorry, I couldn't generate a response."

    # Format as OpenAI response
    return ChatCompletionResponse(
        id=completion_id,
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=final_response),
                finish_reason="stop",
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=0,  # Not tracking for now
            completion_tokens=0,
            total_tokens=0,
        ),
    )


async def stream_chat_completion(
    user_input: str,
    thread_id: str,
    completion_id: str,
    model: str,
) -> AsyncIterator[str]:
    """
    Stream chat completion in OpenAI format.

    Args:
        user_input: User's input message
        thread_id: Thread ID for conversation
        completion_id: Unique completion ID
        model: Model name

    Yields:
        Server-sent events in OpenAI streaming format
    """
    # Send initial chunk with role
    initial_chunk = ChatCompletionChunk(
        id=completion_id,
        model=model,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=ChatCompletionChunkDelta(role="assistant", content=""),
                finish_reason=None,
            )
        ],
    )
    yield f"data: {initial_chunk.model_dump_json()}\n\n"

    # Stream workflow execution
    response_text = ""
    async for event in workflow_service.invoke_workflow(
        user_input=user_input,
        thread_id=thread_id,
        top_k=10,
        stream=True,
    ):
        if event["type"] == "result":
            result = event.get("data", {})
            streamed = False

            # If we have the original prompt, re-run the LLM in streaming mode
            prompt_text = result.get("synthesis_prompt") or result.get(
                "lookup_synthesis_prompt"
            )
            if prompt_text:
                async for chunk in _stream_llm_prompt(
                    prompt_text, completion_id, model
                ):
                    yield chunk
                    streamed = True

            if not streamed:
                # Extract final response (chitchat or fallback)
                if result.get("chitchat_response"):
                    response_text = result.get("chitchat_response")
                elif result.get("final_answer"):
                    response_text = result.get("final_answer")
                else:
                    response_text = "I'm sorry, I couldn't generate a response."

                for content_piece in _chunk_text(response_text):
                    content_chunk = ChatCompletionChunk(
                        id=completion_id,
                        model=model,
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta=ChatCompletionChunkDelta(content=content_piece),
                                finish_reason=None,
                            )
                        ],
                    )
                    yield f"data: {content_chunk.model_dump_json()}\n\n"

    # Send final chunk with finish_reason
    final_chunk = ChatCompletionChunk(
        id=completion_id,
        model=model,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=ChatCompletionChunkDelta(),
                finish_reason="stop",
            )
        ],
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"

    # Send [DONE] message
    yield "data: [DONE]\n\n"


def _chunk_text(text: str, chunk_size: int = 120) -> list[str]:
    """
    Split text into smaller chunks for streaming.

    Args:
        text: Full response text
        chunk_size: Desired chunk size

    Returns:
        List of text chunks
    """
    if not text:
        return [""]

    words = text.split()
    chunks = []
    current = []
    current_len = 0

    for word in words:
        if current_len + len(word) + 1 > chunk_size and current:
            chunks.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += len(word) + 1

    if current:
        chunks.append(" ".join(current))
    return chunks


async def _stream_llm_prompt(
    prompt: str, completion_id: str, model: str
) -> AsyncIterator[str]:
    """
    Stream LLM output for the provided prompt.

    Args:
        prompt: Prompt text to send to the LLM
        completion_id: Completion ID for response chunks
        model: Model name

    Yields:
        Streaming SSE chunks
    """
    llm = ChatOpenAI(
        model=model,
        temperature=settings.temperature,
        streaming=True,
    )

    async for chunk in llm.astream([HumanMessage(content=prompt)]):
        content = getattr(chunk, "content", None)
        if not content:
            continue

        content_chunk = ChatCompletionChunk(
            id=completion_id,
            model=model,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(content=content),
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {content_chunk.model_dump_json()}\n\n"
