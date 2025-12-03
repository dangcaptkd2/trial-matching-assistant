"""Pydantic schemas for API responses."""

import time
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class MessageResponse(BaseModel):
    """Chat message response schema."""

    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Message timestamp"
    )


class ExecutionStep(BaseModel):
    """Single execution step in LangGraph workflow."""

    node_name: str = Field(..., description="Name of the executed node")
    status: str = Field(..., description="Status: 'started', 'completed', 'error'")
    start_time: Optional[datetime] = Field(None, description="Step start time")
    end_time: Optional[datetime] = Field(None, description="Step end time")
    duration_ms: Optional[float] = Field(
        None, description="Execution duration in milliseconds"
    )
    input_summary: Optional[str] = Field(None, description="Summary of node input")
    output_summary: Optional[str] = Field(None, description="Summary of node output")
    error: Optional[str] = Field(None, description="Error message if status is 'error'")


class ExecutionTrace(BaseModel):
    """LangGraph execution trace data."""

    steps: List[ExecutionStep] = Field(
        default_factory=list, description="List of execution steps"
    )
    total_duration_ms: Optional[float] = Field(
        None, description="Total execution time in milliseconds"
    )
    execution_path: List[str] = Field(
        default_factory=list, description="Ordered list of node names in execution path"
    )


class WorkflowResponse(BaseModel):
    """Complete workflow response with trace."""

    response: str = Field(..., description="Final response to user")
    response_type: str = Field(
        ..., description="Type of response: 'chitchat', 'trial_search', 'trial_summary'"
    )
    execution_trace: Optional[ExecutionTrace] = Field(
        None, description="Execution trace data"
    )
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ConversationResponse(BaseModel):
    """Conversation metadata response."""

    conversation_id: str = Field(..., description="Unique conversation/thread ID")
    created_at: datetime = Field(..., description="Conversation creation timestamp")
    last_activity: datetime = Field(..., description="Last activity timestamp")
    message_count: int = Field(..., description="Number of messages in conversation")
    preview: Optional[str] = Field(None, description="Preview of last message")


class ConversationDetailResponse(BaseModel):
    """Detailed conversation response with messages."""

    conversation_id: str = Field(..., description="Unique conversation/thread ID")
    created_at: datetime = Field(..., description="Conversation creation timestamp")
    last_activity: datetime = Field(..., description="Last activity timestamp")
    messages: List[MessageResponse] = Field(
        default_factory=list, description="List of messages"
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="healthy", description="Service status")
    version: str = Field(default="1.0.0", description="API version")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Check timestamp"
    )


# OpenAI API Compatible Schemas


class ChatMessage(BaseModel):
    """Chat message in OpenAI format."""

    role: Literal["system", "user", "assistant"] = Field(
        ..., description="Message role"
    )
    content: str = Field(..., description="Message content")
    name: Optional[str] = Field(None, description="Optional name of the participant")


class ChatCompletionRequest(BaseModel):
    """OpenAI chat completion request."""

    model: str = Field(default="clinical-trial-assistant", description="Model to use")
    messages: List[ChatMessage] = Field(
        ..., description="List of messages in the conversation"
    )
    temperature: Optional[float] = Field(
        0.0, ge=0.0, le=2.0, description="Sampling temperature"
    )
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0, description="Nucleus sampling")
    n: Optional[int] = Field(1, ge=1, le=10, description="Number of completions")
    stream: Optional[bool] = Field(False, description="Whether to stream responses")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    presence_penalty: Optional[float] = Field(0.0, description="Presence penalty")
    frequency_penalty: Optional[float] = Field(0.0, description="Frequency penalty")
    user: Optional[str] = Field(None, description="Unique user identifier")


class ChatCompletionChoice(BaseModel):
    """Single completion choice."""

    index: int = Field(..., description="Choice index")
    message: ChatMessage = Field(..., description="Completion message")
    finish_reason: Literal["stop", "length", "content_filter", "null"] = Field(
        ..., description="Reason for completion finish"
    )


class ChatCompletionUsage(BaseModel):
    """Token usage information."""

    prompt_tokens: int = Field(0, description="Tokens in prompt")
    completion_tokens: int = Field(0, description="Tokens in completion")
    total_tokens: int = Field(0, description="Total tokens")


class ChatCompletionResponse(BaseModel):
    """OpenAI chat completion response."""

    id: str = Field(..., description="Unique completion ID")
    object: Literal["chat.completion"] = Field(
        "chat.completion", description="Object type"
    )
    created: int = Field(
        default_factory=lambda: int(time.time()), description="Creation timestamp"
    )
    model: str = Field(..., description="Model used")
    choices: List[ChatCompletionChoice] = Field(
        ..., description="List of completion choices"
    )
    usage: Optional[ChatCompletionUsage] = Field(None, description="Token usage")


class ChatCompletionChunkDelta(BaseModel):
    """Delta for streaming chunk."""

    role: Optional[Literal["assistant"]] = Field(None, description="Message role")
    content: Optional[str] = Field(None, description="Content delta")


class ChatCompletionChunkChoice(BaseModel):
    """Streaming chunk choice."""

    index: int = Field(..., description="Choice index")
    delta: ChatCompletionChunkDelta = Field(..., description="Content delta")
    finish_reason: Optional[Literal["stop", "length", "content_filter", "null"]] = (
        Field(None, description="Reason for completion finish")
    )


class ChatCompletionChunk(BaseModel):
    """OpenAI streaming chunk."""

    id: str = Field(..., description="Unique completion ID")
    object: Literal["chat.completion.chunk"] = Field(
        "chat.completion.chunk", description="Object type"
    )
    created: int = Field(
        default_factory=lambda: int(time.time()), description="Creation timestamp"
    )
    model: str = Field(..., description="Model used")
    choices: List[ChatCompletionChunkChoice] = Field(..., description="List of choices")


class Model(BaseModel):
    """Model information."""

    id: str = Field(..., description="Model ID")
    object: Literal["model"] = Field("model", description="Object type")
    created: int = Field(
        default_factory=lambda: int(time.time()), description="Creation timestamp"
    )
    owned_by: str = Field("organization", description="Owner organization")


class ModelList(BaseModel):
    """List of available models."""

    object: Literal["list"] = Field("list", description="Object type")
    data: List[Model] = Field(..., description="List of models")
