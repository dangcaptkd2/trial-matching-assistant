"""
Application settings using pydantic_settings.

This module defines all configuration settings for the trial matching system
using Pydantic BaseSettings for type safety and validation.
"""

import os
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

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)

    # API Keys and URLs
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    elasticsearch_url: Optional[str] = "http://localhost:9200"
    # elasticsearch_url: Optional[str] = "http://elasticsearch:9200"

    # LangSmith Settings
    langchain_api_key: Optional[str] = None
    langchain_project: Optional[str] = "clinical-trial-matching"
    langchain_tracing_v2: bool = True

    # LLM Settings
    llm_model: str = "gpt-4.1-nano"
    llm_model_url: Optional[str] = "https://api.openai.com/v1"
    # llm_model: str = "google/medgemma-4b-it"
    # llm_model_url: Optional[str] = (
    #     "https://nb-29498343-695e-406f-afbb-8c20db43d17e-8000-sea1.notebook.console.greennode.ai/v1"
    # )
    temperature: float = 0.0
    llm_judge_model: str = "gpt-5.1"

    # Elasticsearch Settings
    # es_index_name: str = "trec2023_ctnlp"
    es_index_name: str = "aact"

    # PostgreSQL Settings
    postgres_host: str = "localhost"
    # postgres_host: str = "postgres"
    postgres_port: int = 5433
    postgres_database: str = "aact"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    postgres_schema: str = "ctgov"

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    streaming_enabled: bool = False  # Enable/disable streaming responses globally

    # Chainlit Settings
    chainlit_host: Optional[str] = None  # If None, uses FastAPI host
    chainlit_port: Optional[int] = None  # If None, uses FastAPI port


# Create global settings instance
settings = Settings()

# Configure environment variables from settings


# Set OpenAI API key and base URL if provided in settings
if settings.openai_api_key:
    os.environ["OPENAI_API_KEY"] = settings.openai_api_key
if settings.openai_base_url:
    os.environ["OPENAI_BASE_URL"] = settings.openai_base_url

# Configure LangSmith tracing if enabled
if settings.langchain_tracing_v2 and settings.langchain_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    if settings.langchain_project:
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
