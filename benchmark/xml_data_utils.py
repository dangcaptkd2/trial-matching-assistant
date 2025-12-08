"""
Read topics from xml file
"""

import xml.etree.ElementTree as ET
from typing import Any, Dict


def read_topics_xml_file(xml_path: str) -> Dict[str, Any]:
    """
    Read the TREC Clinical Trials topics XML file and extract topic information.

    Args:
        xml_path: Path to the topics XML file

    Returns:
        Dictionary containing topics information with structure:
        {
            "task": str,
            "topics": [
                {
                    "number": str,
                    "content": str  # The clinical case description
                },
                ...
            ]
        }
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        topics_data = {"task": root.get("task", ""), "topics": []}

        # Extract each topic
        for topic in root.findall("topic"):
            topic_info = {
                "number": topic.get("number", ""),
                "content": topic.text.strip() if topic.text else "",
            }

            topics_data["topics"].append(topic_info)

        return topics_data

    except ET.ParseError as e:
        print(f"Error parsing topics XML file {xml_path}: {e}")
        return {}
    except Exception as e:
        print(f"Error reading topics file {xml_path}: {e}")
        return {}


def get_topics_xml_file(
    data_file: str,
) -> Dict[str, Any]:
    """Get all topics from the xml file."""
    topics_data = read_topics_xml_file(data_file)
    return topics_data


def get_topic_by_number(topics_data: Dict[str, Any], topic_number: str) -> Dict[str, Any]:
    """
    Get a specific topic by its number.

    Args:
        topics_data: Topics data from get_topics_xml_file()
        topic_number: The topic number to find

    Returns:
        Dictionary containing the specific topic information with keys:
        - number: str
        - content: str (the clinical case description)
    """
    for topic in topics_data.get("topics", []):
        if topic.get("number") == topic_number:
            return topic
    return {}


def get_topic_content(topics_data: Dict[str, Any], topic_number: str) -> str:
    """
    Get the content (clinical case description) of a specific topic.

    Args:
        topics_data: Topics data from get_topics_xml_file()
        topic_number: The topic number to find

    Returns:
        The clinical case description text, or empty string if not found
    """
    topic = get_topic_by_number(topics_data, topic_number)
    return topic.get("content", "")


def count_topics(topics_data: Dict[str, Any]) -> int:
    """
    Count the total number of topics.

    Args:
        topics_data: Topics data from get_topics_trec_2023()

    Returns:
        Number of topics
    """
    return len(topics_data.get("topics", []))


if __name__ == "__main__":
    # Demo: Load and display topics data
    TOPICS_DIR_TEST = "data/topics2022.xml"
    topics_data_tmp = get_topics_xml_file(TOPICS_DIR_TEST)
    print(topics_data_tmp.keys())
    print(topics_data_tmp["task"])
    print(topics_data_tmp["topics"][0])
    print(len(topics_data_tmp["topics"]))
