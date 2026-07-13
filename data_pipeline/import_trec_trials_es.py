"""Import TREC 2023 ClinicalTrials ZIP files into Elasticsearch using CTnlp parser."""

import argparse
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterator, List

from CTnlp.clinical_trial import ClinicalTrial, Intervention
from CTnlp.parsers import parse_clinical_trials_from_folder
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from loguru import logger

from src.config.settings import settings


DEFAULT_INDEX_NAME = "trec2023_ctnlp"


def load_config(config_path: str = "data_pipeline/config.yaml") -> dict:
    try:
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def connect_elasticsearch(es_url: str) -> Elasticsearch:
    logger.info(f"Connecting to Elasticsearch at {es_url}")
    es = Elasticsearch(
        [es_url],
        timeout=60,
        max_retries=5,
        retry_on_timeout=True,
    )
    if not es.ping():
        raise RuntimeError(f"Cannot connect to Elasticsearch at {es_url}")
    return es


def create_index(es: Elasticsearch, index_name: str, recreate: bool = False) -> None:
    """Create Elasticsearch index with clinical trial mappings."""
    mapping = {
        "mappings": {
            "properties": {
                "nct_id": {"type": "keyword"},
                "identifier": {"type": "keyword"},
                "org_study_id": {"type": "keyword"},
                "brief_title": {"type": "text"},
                "official_title": {"type": "text"},
                "brief_summary": {"type": "text"},
                "detailed_description": {"type": "text"},
                "text": {"type": "text"},
                "eligibility_criteria": {"type": "text"},
                "gender": {"type": "keyword"},
                "minimum_age": {"type": "integer"},
                "maximum_age": {"type": "integer"},
                "accepts_healthy_volunteers": {"type": "boolean"},
                "study_type": {"type": "keyword"},
                "inclusion_list": {"type": "text"},
                "exclusion_list": {"type": "text"},
                "conditions_list": {"type": "text"},
                "interventions_list": {"type": "text"},
                "primary_outcomes_list": {"type": "text"},
                "secondary_outcomes_list": {"type": "text"},
                "condition": {"type": "text"},
                "intervention_type": {"type": "text"},
                "primary_outcome": {"type": "text"},
                "drug_name": {"type": "text"},
                "drug_keywords": {"type": "text"},
                "general_keywords": {"type": "text"},
            }
        }
    }

    if es.indices.exists(index=index_name):
        if recreate:
            logger.info(f"Deleting existing Elasticsearch index: {index_name}")
            es.indices.delete(index=index_name)
        else:
            logger.info(f"Index already exists: {index_name}")
            return

    logger.info(f"Creating Elasticsearch index: {index_name}")
    es.indices.create(index=index_name, body=mapping)


def normalize_ctnlp_trial_for_es(ct: ClinicalTrial) -> Dict[str, Any]:
    """Convert CTnlp ClinicalTrial dataclass to ES document format.

    Args:
        ct: ClinicalTrial dataclass instance from CTnlp

    Returns:
        Dictionary ready for Elasticsearch indexing
    """

    def _as_int(val: float | None) -> int:
        """Convert Optional[float] to int."""
        if val is None:
            return 0
        try:
            return int(val)
        except (ValueError, TypeError):
            return 0

    def _as_str(val: str | None) -> str:
        """Convert Optional[str] to str."""
        return (val or "").strip()

    def _as_list(val: List[str] | None) -> List[str]:
        """Convert Optional[List[str]] to List[str]."""
        if val is None:
            return []
        return [item.strip() for item in val if item and item.strip()]

    def _interventions_to_names(interventions: List[Intervention] | None) -> List[str]:
        """Extract intervention names from Intervention objects."""
        if interventions is None:
            return []
        return [
            interv.name.strip() for interv in interventions if interv and interv.name
        ]

    def _interventions_to_types(interventions: List[Intervention] | None) -> List[str]:
        """Extract intervention types from Intervention objects."""
        if interventions is None:
            return []
        return [
            interv.type.strip() for interv in interventions if interv and interv.type
        ]

    # Extract intervention names and types
    intervention_names = _interventions_to_names(ct.interventions)
    intervention_types = _interventions_to_types(ct.interventions)

    # Build ES document
    doc = {
        # Core identifiers
        "nct_id": _as_str(ct.nct_id),
        "identifier": _as_str(ct.nct_id),
        "org_study_id": _as_str(ct.org_study_id),
        # Titles and descriptions
        "brief_title": _as_str(ct.brief_title),
        "official_title": _as_str(ct.official_title),
        "brief_summary": _as_str(ct.brief_summary),
        "detailed_description": _as_str(ct.detailed_description),
        # Important: text field from ClinicalTrial
        "text": _as_str(ct.text),
        # Eligibility and demographics
        "eligibility_criteria": _as_str(ct.criteria),
        "gender": _as_str(ct.gender.value if ct.gender else None),
        "minimum_age": _as_int(ct.minimum_age),
        "maximum_age": _as_int(ct.maximum_age),
        "accepts_healthy_volunteers": ct.accepts_healthy_volunteers,
        # Study information
        "study_type": _as_str(ct.study_type),
        # Structured lists from CTnlp
        "inclusion_list": "; ".join(_as_list(ct.inclusion)),
        "exclusion_list": "; ".join(_as_list(ct.exclusion)),
        "conditions_list": "; ".join(_as_list(ct.conditions)),
        "interventions_list": "; ".join(intervention_names),
        "primary_outcomes_list": "; ".join(_as_list(ct.primary_outcomes)),
        "secondary_outcomes_list": "; ".join(_as_list(ct.secondary_outcomes)),
        # Legacy fields for backward compatibility
        # Join lists into strings for compatibility with existing search
        "condition": "; ".join(_as_list(ct.conditions)),
        "intervention_type": "; ".join(intervention_types),
        "primary_outcome": "; ".join(_as_list(ct.primary_outcomes)),
        # For drug_name and drug_keywords, use intervention names
        "drug_name": "; ".join(intervention_names),
        "drug_keywords": "",  # Not available in CTnlp ClinicalTrial
        "general_keywords": "",  # Not available in CTnlp ClinicalTrial
    }

    return doc


def generate_es_actions(
    index_name: str,
    trials: List[ClinicalTrial],
) -> Iterator[Dict[str, Any]]:
    """Convert ClinicalTrial objects to ES bulk actions.

    Args:
        index_name: Elasticsearch index name
        trials: List of ClinicalTrial dataclass instances

    Yields:
        ES bulk action dictionaries
    """
    for ct in trials:
        nct_id = (ct.nct_id or "").strip()
        if not nct_id:
            continue

        doc = normalize_ctnlp_trial_for_es(ct)
        yield {
            "_op_type": "index",
            "_index": index_name,
            "_id": nct_id,
            "_source": doc,
        }


def extract_and_parse_zip(zip_path: Path) -> List[ClinicalTrial]:
    """Extract ZIP file and parse clinical trials using CTnlp.

    Args:
        zip_path: Path to ZIP file containing XML clinical trial files

    Returns:
        List of parsed ClinicalTrial objects
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        logger.info(f"Extracting {zip_path} to temporary directory")
        
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(temp_dir_path)
        
        logger.info(f"Parsing clinical trials from extracted files using CTnlp")
        try:
            trials = parse_clinical_trials_from_folder(folder_name=str(temp_dir_path))
            logger.info(f"Parsed {len(trials)} clinical trials from {zip_path}")
            return trials
        except Exception as exc:
            logger.error(f"Failed to parse trials from {zip_path}: {exc}")
            raise


def index_documents(
    es: Elasticsearch,
    index_name: str,
    trials: List[ClinicalTrial],
    chunk_size: int = 1000,
) -> int:
    """Index clinical trials into Elasticsearch.

    Args:
        es: Elasticsearch client
        index_name: Target index name
        trials: List of ClinicalTrial objects to index
        chunk_size: Number of documents per bulk request

    Returns:
        Number of successfully indexed documents
    """
    logger.info(f"Indexing {len(trials)} trials into {index_name} with chunk size {chunk_size}")
    success, failed = bulk(
        es,
        generate_es_actions(index_name, trials),
        chunk_size=chunk_size,
        raise_on_error=False,
        request_timeout=60,
    )
    
    if failed:
        logger.error(f"Bulk indexing completed with {len(failed)} failures")
    
    return success


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import TREC 2023 ClinicalTrials data into Elasticsearch using CTnlp"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("./data/trec_trials"),
        help="Directory containing downloaded TREC ClinicalTrials ZIP files",
    )
    parser.add_argument(
        "--index-name",
        type=str,
        default=DEFAULT_INDEX_NAME,
        help="Elasticsearch index name to write documents into",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="data_pipeline/config.yaml",
        help="Path to pipeline configuration file",
    )
    parser.add_argument(
        "--create-index",
        action="store_true",
        help="Create the Elasticsearch index before importing (will not delete an existing index unless --recreate-index is also provided)",
    )
    parser.add_argument(
        "--recreate-index",
        action="store_true",
        help="Delete and recreate the Elasticsearch index before importing",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Bulk chunk size for Elasticsearch indexing",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    index_name = args.index_name
    if not args.index_name and config.get("trec_trials", {}).get("index_name"):
        index_name = config["trec_trials"]["index_name"]

    if not index_name:
        index_name = DEFAULT_INDEX_NAME

    es_url = settings.elasticsearch_url or "http://localhost:9200"
    es = connect_elasticsearch(es_url)

    if args.create_index or args.recreate_index:
        create_index(es, index_name, recreate=args.recreate_index)
    else:
        if not es.indices.exists(index=index_name):
            logger.info(f"Index '{index_name}' does not exist, creating it")
            create_index(es, index_name, recreate=False)

    input_dir = args.input_dir
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return 1

    zip_files = sorted([path for path in input_dir.glob("*.zip")])
    if not zip_files:
        logger.error(f"No ZIP files found in {input_dir}")
        return 1

    total_success = 0
    total_trials = 0
    for zip_path in zip_files:
        logger.info(f"Processing archive: {zip_path}")
        try:
            trials = extract_and_parse_zip(zip_path)
            if not trials:
                logger.warning(f"No trials parsed from {zip_path}")
                continue

            total_trials += len(trials)
            success = index_documents(es, index_name, trials, chunk_size=args.chunk_size)
            total_success += success
            logger.success(f"Indexed {success} / {len(trials)} trials from {zip_path}")
        except Exception as exc:
            logger.error(f"Failed to process {zip_path}: {exc}")
            continue

    if total_trials == 0:
        logger.error("No trials were parsed from any ZIP files")
        return 1

    logger.success(f"✓ Finished importing {total_success} trials into '{index_name}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
