"""Import TREC 2023 ClinicalTrials ZIP files into Elasticsearch."""

import argparse
import json
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from loguru import logger

from src.config.settings import settings

DEFAULT_INDEX_NAME = "trec2023_ctnlp"
SUPPORTED_JSON_EXTENSIONS = {".json", ".jsonl", ".ndjson"}
SUPPORTED_XML_EXTENSIONS = {".xml"}


def load_config(config_path: str = "data_pipeline/config.yaml") -> dict:
    try:
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def connect_elasticsearch(es_url: str) -> Elasticsearch:
    logger.info(f"Connecting to Elasticsearch at {es_url}")
    es = Elasticsearch([es_url])
    if not es.ping():
        raise RuntimeError(f"Cannot connect to Elasticsearch at {es_url}")
    return es


def create_index(es: Elasticsearch, index_name: str, recreate: bool = False) -> None:
    mapping = {
        "mappings": {
            "properties": {
                "nct_id": {"type": "keyword"},
                "title": {"type": "text"},
                "official_title": {"type": "text"},
                "brief_summary": {"type": "text"},
                "conditions": {"type": "text"},
                "interventions": {"type": "text"},
                "keywords": {"type": "text"},
                "eligibility_criteria": {"type": "text"},
                "countries": {"type": "keyword"},
                "cities": {"type": "keyword"},
                "sites": {"type": "text"},
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


def normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: normalize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_value(v) for v in value]
    return value


def normalize_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {k: normalize_value(v) for k, v in doc.items()}
    if "id" in normalized and "nct_id" not in normalized:
        normalized["nct_id"] = normalized["id"]
    return normalized


def parse_json_file(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [normalize_document(doc) for doc in data if isinstance(doc, dict)]

    if isinstance(data, dict):
        if "trials" in data and isinstance(data["trials"], list):
            return [normalize_document(doc) for doc in data["trials"] if isinstance(doc, dict)]
        if "studies" in data and isinstance(data["studies"], list):
            return [normalize_document(doc) for doc in data["studies"] if isinstance(doc, dict)]
        if "clinical_study" in data and isinstance(data["clinical_study"], list):
            return [normalize_document(doc) for doc in data["clinical_study"] if isinstance(doc, dict)]
        return [normalize_document(data)]

    return []


def parse_json_lines_file(path: Path) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                if isinstance(doc, dict):
                    docs.append(normalize_document(doc))
            except json.JSONDecodeError as exc:
                logger.warning(f"Skipping invalid JSON line in {path}: {exc}")
    return docs


def parse_xml_file(path: Path) -> List[Dict[str, Any]]:
    """Parse TREC clinical trial XML file into a normalized document."""
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        
        # Extract trial data from XML
        doc = {}
        
        # Get NCT ID
        nct_id_elem = root.find(".//nct_id")
        if nct_id_elem is not None and nct_id_elem.text:
            doc["nct_id"] = nct_id_elem.text.strip()
        
        # Get titles
        title_elem = root.find(".//official_title")
        if title_elem is not None and title_elem.text:
            doc["official_title"] = title_elem.text.strip()
        else:
            brief_title_elem = root.find(".//brief_title")
            if brief_title_elem is not None and brief_title_elem.text:
                doc["official_title"] = brief_title_elem.text.strip()
        
        # Get brief summary
        brief_summary_elem = root.find(".//brief_summary/textblock")
        if brief_summary_elem is not None and brief_summary_elem.text:
            doc["brief_summary"] = brief_summary_elem.text.strip()
        
        # Get conditions
        conditions = []
        for condition_elem in root.findall(".//condition"):
            if condition_elem.text:
                conditions.append(condition_elem.text.strip())
        if conditions:
            doc["conditions"] = " ".join(conditions)
        
        # Get interventions
        interventions = []
        for intervention_elem in root.findall(".//intervention_name"):
            if intervention_elem.text:
                interventions.append(intervention_elem.text.strip())
        if interventions:
            doc["interventions"] = " ".join(interventions)
        
        # Get keywords
        keywords = []
        for keyword_elem in root.findall(".//keyword"):
            if keyword_elem.text:
                keywords.append(keyword_elem.text.strip())
        if keywords:
            doc["keywords"] = " ".join(keywords)
        
        # Get eligibility criteria
        criteria_elem = root.find(".//eligibility/criteria/textblock")
        if criteria_elem is not None and criteria_elem.text:
            doc["eligibility_criteria"] = criteria_elem.text.strip()
        
        # Get locations (countries, cities)
        countries = set()
        cities = set()
        for facility_elem in root.findall(".//facility"):
            country_elem = facility_elem.find("country")
            if country_elem is not None and country_elem.text:
                countries.add(country_elem.text.strip())
            city_elem = facility_elem.find("city")
            if city_elem is not None and city_elem.text:
                cities.add(city_elem.text.strip())
        
        if countries:
            doc["countries"] = list(countries)
        if cities:
            doc["cities"] = list(cities)
        
        if doc.get("nct_id"):
            return [normalize_document(doc)]
        return []
    
    except ET.ParseError as exc:
        logger.warning(f"Failed to parse XML file {path}: {exc}")
        return []
    except Exception as exc:
        logger.warning(f"Error processing XML file {path}: {exc}")
        return []


def read_documents_from_file(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return parse_json_file(path)
    if suffix in {".jsonl", ".ndjson"}:
        return parse_json_lines_file(path)
    if suffix in SUPPORTED_XML_EXTENSIONS:
        return parse_xml_file(path)

    logger.warning(f"Skipping unsupported file type: {path}")
    return []


def generate_es_actions(documents: Iterator[Dict[str, Any]], index_name: str) -> Iterator[Dict[str, Any]]:
    for doc in documents:
        action = {
            "_op_type": "index",
            "_index": index_name,
            "_source": doc,
        }
        if doc.get("nct_id"):
            action["_id"] = doc["nct_id"]
        yield action


def extract_documents_from_zip(zip_path: Path) -> Iterator[Dict[str, Any]]:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        with zipfile.ZipFile(zip_path, "r") as archive:
            members = [m for m in archive.namelist() if not m.endswith("/")]
            for member in members:
                archive.extract(member, temp_dir_path)
                candidate = temp_dir_path / member
                # Support both JSON and XML files inside the archive
                if candidate.suffix.lower() in (SUPPORTED_JSON_EXTENSIONS | SUPPORTED_XML_EXTENSIONS):
                    for doc in read_documents_from_file(candidate):
                        yield doc
                else:
                    logger.warning(f"Unsupported file inside ZIP: {member}")


def index_documents(es: Elasticsearch, index_name: str, documents: Iterator[Dict[str, Any]], chunk_size: int = 1000) -> int:
    logger.info(f"Indexing documents into {index_name} with chunk size {chunk_size}")
    success, failed = bulk(es, generate_es_actions(documents, index_name), chunk_size=chunk_size, raise_on_error=False)
    if failed:
        logger.error(f"Bulk indexing completed with {len(failed)} failures")
    return success


def main() -> int:
    parser = argparse.ArgumentParser(description="Import TREC 2023 ClinicalTrials data into Elasticsearch")
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
    total_docs = 0
    for zip_path in zip_files:
        logger.info(f"Processing archive: {zip_path}")
        docs = list(extract_documents_from_zip(zip_path))
        count = len(docs)
        if count == 0:
            logger.warning(f"No supported documents found in {zip_path}")
            continue

        total_docs += count
        success = index_documents(es, index_name, iter(docs), chunk_size=args.chunk_size)
        total_success += success
        logger.success(f"Indexed {success} / {count} documents from {zip_path}")

    if total_docs == 0:
        logger.error("No documents were indexed")
        return 1

    logger.success(f"✓ Finished importing {total_success} documents into '{index_name}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
