"""Memory and checkpoint configuration for LangGraph."""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver


def get_checkpointer() -> BaseCheckpointSaver:
    """Get checkpoint store based on settings.

    Returns:
        Checkpoint store instance (MemorySaver for dev, PostgresSaver for prod)
    """
    # For now, use MemorySaver
    # In production, you can switch to PostgresSaver:
    # if settings.use_postgres_checkpoint:
    #     from langgraph.checkpoint.postgres import PostgresSaver
    #     return PostgresSaver.from_conn_string(settings.postgres_url)
    return MemorySaver()
