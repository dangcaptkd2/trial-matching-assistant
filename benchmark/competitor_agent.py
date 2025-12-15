#!/usr/bin/env python3
"""Simple GPT agent that does direct LLM calls."""

import json

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from src.config.settings import settings


async def gpt_invoke(user_input: str) -> dict:
    """
    Simple GPT agent that does a direct LLM call.

    Args:
        user_input: The query string

    Returns:
        dict with same structure as WorkflowService result:
        - final_answer: text response from GPT
        - reranked_results: empty list (no search performed)
    """
    llm = ChatOpenAI(model=settings.llm_model, temperature=settings.temperature)
    response = await llm.ainvoke([HumanMessage(content=user_input)])

    return response.content


def load_dataset(dataset_file: str) -> list:
    """Load topics from dataset JSON file."""

    with open(dataset_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("topics", [])
