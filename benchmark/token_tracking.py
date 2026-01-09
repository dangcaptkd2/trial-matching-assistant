"""Token usage tracking utilities for benchmark evaluation."""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class TokenUsage:
    """Token usage information."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Add two TokenUsage instances together."""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


def extract_tokens_from_response(response: Any) -> TokenUsage:
    """
    Extract token usage from a LangChain LLM response.

    Args:
        response: LangChain response object (AIMessage or similar)

    Returns:
        TokenUsage object with extracted token counts
    """
    # Try to get usage_metadata (newer LangChain versions)
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        metadata = response.usage_metadata
        return TokenUsage(
            prompt_tokens=metadata.get("input_tokens", 0),
            completion_tokens=metadata.get("output_tokens", 0),
            total_tokens=metadata.get("total_tokens", 0),
        )

    # Try response_metadata (older format)
    if hasattr(response, "response_metadata") and response.response_metadata:
        token_usage = response.response_metadata.get("token_usage", {})
        return TokenUsage(
            prompt_tokens=token_usage.get("prompt_tokens", 0),
            completion_tokens=token_usage.get("completion_tokens", 0),
            total_tokens=token_usage.get("total_tokens", 0),
        )

    # Return empty usage if not found
    return TokenUsage()


def aggregate_tokens(token_list: list[TokenUsage]) -> TokenUsage:
    """
    Aggregate multiple TokenUsage objects.

    Args:
        token_list: List of TokenUsage objects

    Returns:
        Aggregated TokenUsage
    """
    result = TokenUsage()
    for tokens in token_list:
        result = result + tokens
    return result
