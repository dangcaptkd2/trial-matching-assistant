#!/usr/bin/env python3
"""Translate topics content to Vietnamese using LLM judger model."""

import asyncio
import json
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from benchmark.xml_data_utils import get_topics_xml_file
from src.config.settings import settings

TRANSLATION_PROMPT = """You are a professional translator. Translate the following English text to Vietnamese. 
Maintain the medical terminology accuracy and preserve the original meaning.

English text:
{content}

Provide only the Vietnamese translation, without any additional explanation or commentary."""


async def translate_content(content: str, llm: ChatOpenAI) -> str:
    """
    Translate content to Vietnamese using LLM.

    Args:
        content: English content to translate
        llm: LLM instance for translation

    Returns:
        Vietnamese translation
    """
    if not content or not content.strip():
        return ""

    prompt = TRANSLATION_PROMPT.format(content=content)

    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        translation = response.content.strip()

        # Clean up if there are any markdown code blocks
        if "```" in translation:
            # Extract content from code blocks if present
            if "```vietnamese" in translation.lower() or "```vi" in translation.lower():
                translation = translation.split("```")[1].split("```")[0].strip()
                # Remove language identifier if present
                if translation.startswith("vietnamese") or translation.startswith("vi"):
                    translation = translation.split("\n", 1)[-1].strip()
            elif "```" in translation:
                parts = translation.split("```")
                if len(parts) >= 3:
                    translation = parts[1].split("\n", 1)[-1].strip() if "\n" in parts[1] else parts[1].strip()

        return translation
    except Exception as e:
        print(f"  Translation error: {str(e)}")
        return f"[Translation error: {str(e)}]"


async def translate_topics(topics_data: dict, output_file: str):
    """
    Translate all topics to Vietnamese and save to JSON.

    Args:
        topics_data: Topics data from get_topics_xml_file()
        output_file: Path to output JSON file
    """
    # Initialize LLM
    llm = ChatOpenAI(model=settings.llm_judge_model, temperature=0.0)

    translated_topics = []
    total_topics = len(topics_data.get("topics", []))

    print(f"Translating {total_topics} topics to Vietnamese...")
    print(f"Using LLM model: {settings.llm_judge_model}")
    print("-" * 80)

    for idx, topic in enumerate(topics_data.get("topics", []), 1):
        topic_number = topic.get("number", "")
        content = topic.get("content", "")

        print(f"[{idx}/{total_topics}] Translating topic {topic_number}...")

        # Translate content
        vi_content = await translate_content(content, llm)

        translated_topic = {"number": topic_number, "content": content, "vi_content": vi_content}

        translated_topics.append(translated_topic)
        print(f"  ✓ Completed topic {topic_number}")

        # Small delay to avoid rate limiting
        await asyncio.sleep(0.5)

    # Save to JSON
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(translated_topics, f, ensure_ascii=False, indent=2)

    print("-" * 80)
    print(f"✓ Translation complete! Saved {len(translated_topics)} topics to {output_file}")


async def main():
    """Main function to run translation."""
    # Default paths
    topics_file = "data/topics2022.xml"
    output_file = "benchmark/results/translated_topics.json"

    # Allow command line arguments
    if len(sys.argv) > 1:
        topics_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    print("=" * 80)
    print("Topic Translation Script")
    print("=" * 80)
    print(f"Input file: {topics_file}")
    print(f"Output file: {output_file}")
    print()

    # Load topics
    print("Loading topics from XML file...")
    topics_data = get_topics_xml_file(topics_file)

    if not topics_data or not topics_data.get("topics"):
        print(f"Error: No topics found in {topics_file}")
        sys.exit(1)

    print(f"✓ Loaded {len(topics_data.get('topics', []))} topics")
    print()

    # Translate topics
    await translate_topics(topics_data, output_file)


if __name__ == "__main__":
    asyncio.run(main())
