"""
Application settings using pydantic_settings.

This module defines all configuration settings for the trial matching system
using Pydantic BaseSettings for type safety and validation.
"""

from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env file into os.environ BEFORE any other imports
# This ensures Langfuse and other SDKs can read environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False
    )

    # API Keys and URLs
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    elasticsearch_url: Optional[str] = None

    # LangSmith Settings
    langchain_api_key: Optional[str] = None
    langchain_project: Optional[str] = "clinical-trial-matching"
    langchain_tracing_v2: bool = True

    # LLM Settings
    llm_model: str = "gpt-4.1-nano"
    temperature: float = 0.0

    # Elasticsearch Settings
    es_index_name: str = "trec2023_ctnlp"

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    streaming_enabled: bool = True  # Enable/disable streaming responses globally

    # Chainlit Settings
    chainlit_host: Optional[str] = None  # If None, uses FastAPI host
    chainlit_port: Optional[int] = None  # If None, uses FastAPI port


# Create global settings instance
settings = Settings()

# Configure LangSmith tracing if enabled
if settings.langchain_tracing_v2 and settings.langchain_api_key:
    import os

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    if settings.langchain_project:
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
