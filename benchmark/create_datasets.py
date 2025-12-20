#!/usr/bin/env python3
"""Simple script to create datasets for 6 benchmark experiments."""

import json
import random
from pathlib import Path
from typing import Dict, List

from benchmark.xml_data_utils import (
    get_topics_xml_file,
    parse_qrels_file,
    read_topics2023_xml_file,
    sample_qrels_by_label,
    sample_trials_for_comparison,
)
from src.config.settings import settings
from src.services.es_search import ElasticsearchTrialSearcher

# Trial IDs for summarization dataset (from 02_trial_summarize.py)
TRIAL_IDS = [
    "NCT02132585",
    "NCT03341637",
    "NCT00480792",
    "NCT01079793",
    "NCT01670110",
    "NCT00966875",
    "NCT03087305",
    "NCT04888793",
    "NCT01815333",
    "NCT00399217",
    "NCT03966742",
    "NCT00601666",
    "NCT05679622",
    "NCT03701048",
    "NCT01757743",
    "NCT01711658",
    "NCT04912570",
    "NCT04130009",
    "NCT02152072",
    "NCT03835286",
    "NCT05316311",
    "NCT01876264",
    "NCT00679497",
    "NCT00291018",
    "NCT04498546",
    "NCT05611359",
    "NCT02967120",
    "NCT04541875",
    "NCT00250354",
    "NCT05851157",
    "NCT04086849",
    "NCT05735821",
    "NCT01478932",
    "NCT05758727",
    "NCT01061502",
    "NCT02708654",
    "NCT01622413",
    "NCT00473707",
    "NCT01940198",
    "NCT00274963",
    "NCT00005646",
    "NCT00004833",
    "NCT02047695",
    "NCT01301989",
    "NCT03061903",
    "NCT05153993",
    "NCT00195767",
    "NCT00213486",
    "NCT00485602",
    "NCT05146674",
]

# Default queries for translate terms (from 05_trial_translate_terms.py)
DEFAULT_QUERIES_EN = [
    "What does 'progression-free survival' mean?",
    "Explain EGFR mutation",
    "What do these terms mean: adjuvant therapy and metastasis?",
    "What is chemotherapy?",
    "Explain what 'overall survival' means",
    "What does 'biopsy' mean in cancer diagnosis?",
    "Can you explain what 'tumor marker' means?",
    "What is the meaning of 'immunotherapy'?",
    "What does 'staging' mean in cancer?",
    "Explain what 'remission' means in medical terms",
]

DEFAULT_QUERIES_VI = [
    "'Progression-free survival' có nghĩa là gì?",
    "Giải thích đột biến EGFR",
    "Các thuật ngữ này có nghĩa là gì: liệu pháp bổ trợ và di căn?",
    "Hóa trị là gì?",
    "Giải thích 'overall survival' có nghĩa là gì",
    "'Biopsy' có nghĩa là gì trong chẩn đoán ung thư?",
    "Bạn có thể giải thích 'tumor marker' có nghĩa là gì không?",
    "Ý nghĩa của 'immunotherapy' là gì?",
    "'Staging' có nghĩa là gì trong ung thư?",
    "Giải thích 'remission' có nghĩa là gì trong thuật ngữ y tế",
]


def create_matching_dataset(
    topics_file: str = "data/topics2022.xml",
    output_file: str = "benchmark/datasets/01_matching_dataset.json",
) -> str:
    """Create dataset for trial matching experiment."""
    print(f"Creating matching dataset from {topics_file}...")

    try:
        topics_data = get_topics_xml_file(topics_file)
        topics = topics_data.get("topics", [])

        # Load Vietnamese translations if available
        translated_file = "benchmark/results/translated_topics.json"
        translated_dict = {}
        try:
            with open(translated_file, "r", encoding="utf-8") as f:
                translated_data = json.load(f)
                translated_dict = {item["number"]: item for item in translated_data}
        except FileNotFoundError:
            print(f"  Warning: Translated topics file not found: {translated_file}")

        # Merge translated content
        for topic in topics:
            topic_number = topic["number"]
            if topic_number in translated_dict:
                topic["vi_content"] = translated_dict[topic_number].get("vi_content", "")

        dataset = {
            "experiment": "trial_matching",
            "source_file": topics_file,
            "total_topics": len(topics),
            "topics": topics,
        }

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        print(f"  ✓ Created {output_file} with {len(topics)} topics")
        return str(output_path)

    except Exception as e:
        print(f"  ✗ Error creating matching dataset: {str(e)}")
        return ""


def create_summarize_dataset(
    output_file: str = "benchmark/datasets/02_summarize_dataset.json",
) -> str:
    """Create dataset for trial summarization experiment."""
    print("Creating summarize dataset...")

    try:
        dataset = {
            "experiment": "trial_summarization",
            "total_trials": len(TRIAL_IDS),
            "trial_ids": TRIAL_IDS,
        }

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        print(f"  ✓ Created {output_file} with {len(TRIAL_IDS)} trial IDs")
        return str(output_path)

    except Exception as e:
        print(f"  ✗ Error creating summarize dataset: {str(e)}")
        return ""


def create_eligibility_dataset(
    topics_file: str = "/Users/quyen.nguyen/Personal/uit/trec_2023/topics2023.xml",
    qrels_file: str = "data/qrels2022.txt",
    output_file: str = "benchmark/datasets/03_eligibility_dataset.json",
    label_counts: Dict[int, int] = None,
) -> str:
    """Create dataset for eligibility check experiment."""
    print(f"Creating eligibility dataset from {topics_file} and {qrels_file}...")

    if label_counts is None:
        label_counts = {0: 17, 1: 17, 2: 16}

    try:
        # Load topics
        topics_data = read_topics2023_xml_file(topics_file)
        topics_dict = {topic["number"]: topic for topic in topics_data.get("topics", [])}
        print(f"  Loaded {len(topics_dict)} topics")

        # Load qrels
        qrels = parse_qrels_file(qrels_file)
        print(f"  Loaded {len(qrels)} qrels entries")

        # Sample qrels by label
        samples = sample_qrels_by_label(qrels, label_counts, seed=42)
        print(f"  Sampled {len(samples)} topic-trial pairs")

        # Build dataset with patient profiles
        dataset_samples = []
        for topic_number, trial_id, label in samples:
            topic = topics_dict.get(topic_number)
            if topic:
                dataset_samples.append(
                    {
                        "topic_number": topic_number,
                        "trial_id": trial_id,
                        "ground_truth_label": label,
                        "patient_profile": topic.get("content", ""),
                    }
                )

        dataset = {
            "experiment": "eligibility_check",
            "source_files": {
                "topics": topics_file,
                "qrels": qrels_file,
            },
            "label_counts": label_counts,
            "total_samples": len(dataset_samples),
            "samples": dataset_samples,
        }

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        print(f"  ✓ Created {output_file} with {len(dataset_samples)} samples")
        return str(output_path)

    except Exception as e:
        print(f"  ✗ Error creating eligibility dataset: {str(e)}")
        return ""


def create_explain_criteria_dataset(
    output_file: str = "benchmark/datasets/04_explain_criteria_dataset.json",
    count: int = 10,
    fallback_trial_ids: List[str] = None,
) -> str:
    """Create dataset for explain criteria experiment."""
    print("Creating explain criteria dataset...")

    if fallback_trial_ids is None:
        # Use first N trial IDs from summarize dataset as fallback
        fallback_trial_ids = TRIAL_IDS[:count]

    try:
        # Try to get random trial IDs from Elasticsearch
        trial_ids = None
        try:
            es_searcher = ElasticsearchTrialSearcher(index_name=settings.es_index_name)

            # Use match_all query to get a sample
            body = {
                "size": min(count * 3, 10000),
                "query": {"match_all": {}},
                "_source": ["nct_id"],
            }

            resp = es_searcher.es.client.search(index=es_searcher.index_name, body=body)
            hits = resp.get("hits", {}).get("hits", [])

            # Extract trial IDs
            available_ids = []
            for hit in hits:
                nct_id = hit.get("_id") or hit.get("_source", {}).get("nct_id")
                if nct_id:
                    available_ids.append(nct_id)

            if len(available_ids) >= count:
                random.seed(42)
                trial_ids = random.sample(available_ids, count)
                print(f"  Retrieved {len(trial_ids)} random trial IDs from Elasticsearch")
            else:
                print(f"  Warning: Only {len(available_ids)} trials found in Elasticsearch, using fallback")
                trial_ids = fallback_trial_ids

        except Exception as e:
            print(f"  Warning: Could not connect to Elasticsearch: {str(e)}")
            print("  Using fallback trial IDs")
            trial_ids = fallback_trial_ids

        dataset = {
            "experiment": "explain_criteria",
            "source": "elasticsearch" if trial_ids != fallback_trial_ids else "fallback",
            "total_trials": len(trial_ids),
            "trial_ids": trial_ids,
        }

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        print(f"  ✓ Created {output_file} with {len(trial_ids)} trial IDs")
        return str(output_path)

    except Exception as e:
        print(f"  ✗ Error creating explain criteria dataset: {str(e)}")
        return ""


def create_translate_terms_dataset(
    output_file: str = "benchmark/datasets/05_translate_terms_dataset.json",
    use_defaults: bool = True,
) -> str:
    """Create dataset for translate terms experiment."""
    print("Creating translate terms dataset...")

    try:
        queries_en = DEFAULT_QUERIES_EN.copy()
        queries_vi = DEFAULT_QUERIES_VI.copy()

        if not use_defaults:
            print("  Enter English queries (one per line, empty line to finish):")
            queries_en = []
            while True:
                query = input("  > ").strip()
                if not query:
                    break
                queries_en.append(query)

            print("  Enter Vietnamese queries (one per line, empty line to finish):")
            queries_vi = []
            while True:
                query = input("  > ").strip()
                if not query:
                    break
                queries_vi.append(query)

        dataset = {
            "experiment": "translate_terms",
            "total_queries_en": len(queries_en),
            "total_queries_vi": len(queries_vi),
            "queries_en": queries_en,
            "queries_vi": queries_vi,
        }

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        print(f"  ✓ Created {output_file} with {len(queries_en)} EN and {len(queries_vi)} VI queries")
        return str(output_path)

    except Exception as e:
        print(f"  ✗ Error creating translate terms dataset: {str(e)}")
        return ""


def create_compare_trials_dataset(
    qrels_file: str = "data/qrels2022.txt",
    output_file: str = "benchmark/datasets/06_compare_trials_dataset.json",
    num_topics: int = 10,
) -> str:
    """Create dataset for compare trials experiment."""
    print(f"Creating compare trials dataset from {qrels_file}...")

    try:
        # Load qrels
        qrels = parse_qrels_file(qrels_file)
        print(f"  Loaded {len(qrels)} qrels entries")

        # Sample topics with 2-3 eligible trials each
        samples = sample_trials_for_comparison(qrels, num_topics=num_topics, min_trials=2, max_trials=3, seed=42)
        print(f"  Sampled {len(samples)} topic groups")

        dataset_samples = []
        for topic_number, trial_ids in samples:
            dataset_samples.append(
                {
                    "topic_number": topic_number,
                    "trial_ids": trial_ids,
                }
            )

        dataset = {
            "experiment": "compare_trials",
            "source_file": qrels_file,
            "num_topics": num_topics,
            "total_samples": len(dataset_samples),
            "samples": dataset_samples,
        }

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        print(f"  ✓ Created {output_file} with {len(dataset_samples)} samples")
        return str(output_path)

    except Exception as e:
        print(f"  ✗ Error creating compare trials dataset: {str(e)}")
        return ""


def main():
    """Main function to create all datasets."""
    print("=" * 80)
    print("Creating Datasets for 6 Benchmark Experiments")
    print("=" * 80)
    print()

    results = []

    # Create all datasets
    results.append(("01_matching", create_matching_dataset()))
    results.append(("02_summarize", create_summarize_dataset()))
    results.append(("03_eligibility", create_eligibility_dataset()))
    results.append(("04_explain_criteria", create_explain_criteria_dataset()))
    results.append(("05_translate_terms", create_translate_terms_dataset(use_defaults=True)))
    results.append(("06_compare_trials", create_compare_trials_dataset()))

    # Summary
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)

    successful = [name for name, path in results if path]
    failed = [name for name, path in results if not path]

    if successful:
        print(f"\n✓ Successfully created {len(successful)} datasets:")
        for name, path in results:
            if path:
                print(f"  - {name}: {path}")

    if failed:
        print(f"\n✗ Failed to create {len(failed)} datasets:")
        for name, path in results:
            if not path:
                print(f"  - {name}")

    print("=" * 80)


if __name__ == "__main__":
    main()
