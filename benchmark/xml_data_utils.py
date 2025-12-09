"""
Read topics from xml file
"""

import random
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Tuple


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


def read_topics2023_xml_file(xml_path: str) -> Dict[str, Any]:
    """
    Read the TREC Clinical Trials topics2023 XML file with structured fields.

    Args:
        xml_path: Path to the topics2023 XML file

    Returns:
        Dictionary containing topics information with structure:
        {
            "task": str,
            "topics": [
                {
                    "number": str,
                    "template": str,
                    "content": str  # Natural language patient profile built from fields
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
            topic_number = topic.get("number", "")
            template = topic.get("template", "")

            # Build patient profile from fields
            fields = []
            for field in topic.findall("field"):
                field_name = field.get("name", "")
                field_value = field.text.strip() if field.text else ""

                # Skip empty fields
                if not field_value:
                    continue

                # Format field as "field_name: field_value"
                fields.append(f"{field_name}: {field_value}")

            # Combine fields into natural language patient profile
            patient_profile = ", ".join(fields)

            topic_info = {
                "number": topic_number,
                "template": template,
                "content": patient_profile,
            }

            topics_data["topics"].append(topic_info)

        return topics_data

    except ET.ParseError as e:
        print(f"Error parsing topics2023 XML file {xml_path}: {e}")
        return {}
    except Exception as e:
        print(f"Error reading topics2023 file {xml_path}: {e}")
        return {}


def parse_qrels_file(qrels_path: str) -> Dict[Tuple[str, str], int]:
    """
    Parse qrels file and return mapping of (topic_number, trial_id) to label.

    Args:
        qrels_path: Path to the qrels file

    Returns:
        Dictionary mapping (topic_number, trial_id) to label (0, 1, or 2)
        Format: {(topic_number, trial_id): label}
    """
    qrels = {}

    try:
        with open(qrels_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) >= 4:
                    topic_number = parts[0]
                    iteration = parts[1]  # Usually "0", not used
                    trial_id = parts[2]
                    label = int(parts[3])

                    qrels[(topic_number, trial_id)] = label

        return qrels

    except Exception as e:
        print(f"Error parsing qrels file {qrels_path}: {e}")
        return {}


def sample_qrels_by_label(
    qrels: Dict[Tuple[str, str], int], label_counts: Dict[int, int], seed: int = None
) -> List[Tuple[str, str, int]]:
    """
    Randomly sample qrels entries by label.

    Args:
        qrels: Dictionary mapping (topic_number, trial_id) to label
        label_counts: Dictionary mapping label to desired count
            e.g., {0: 17, 1: 17, 2: 16}
        seed: Random seed for reproducibility

    Returns:
        List of tuples: [(topic_number, trial_id, label), ...]
    """
    if seed is not None:
        random.seed(seed)

    # Group qrels by label
    by_label: Dict[int, List[Tuple[str, str]]] = {}
    for (topic_number, trial_id), label in qrels.items():
        if label not in by_label:
            by_label[label] = []
        by_label[label].append((topic_number, trial_id))

    # Sample from each label
    samples = []
    for label, count in label_counts.items():
        if label not in by_label:
            print(f"Warning: No entries found for label {label}")
            continue

        available = by_label[label]
        if len(available) < count:
            print(f"Warning: Only {len(available)} entries available for label {label}, requested {count}")
            count = len(available)

        sampled = random.sample(available, count)
        for topic_number, trial_id in sampled:
            samples.append((topic_number, trial_id, label))

    return samples


if __name__ == "__main__":
    # Demo: Load and display topics data
    TOPICS_DIR_TEST = "data/topics2022.xml"
    topics_data_tmp = get_topics_xml_file(TOPICS_DIR_TEST)
    print(topics_data_tmp.keys())
    print(topics_data_tmp["task"])
    print(topics_data_tmp["topics"][0])
    print(len(topics_data_tmp["topics"]))
