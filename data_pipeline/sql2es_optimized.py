import time
from typing import List, Dict, Any, Tuple

import psycopg2
import psycopg2.extras
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, streaming_bulk
from loguru import logger

from src.config.settings import settings

# PostgreSQL connection parameters
pg_conn_params = {
    "host": settings.postgres_host,
    "port": settings.postgres_port,
    "dbname": settings.postgres_database,
    "user": settings.postgres_user,
    "password": settings.postgres_password,
}

# Elasticsearch connection
es = Elasticsearch([settings.elasticsearch_url])
index_name = settings.es_index_name

# Elasticsearch index mapping
mapping = {
    "mappings": {
        "properties": {
            "nct_id": {"type": "keyword"},
            "brief_title": {"type": "text"},
            "official_title": {"type": "text"},
            "brief_summary": {"type": "text"},
            "conditions": {"type": "text"},
            "interventions": {"type": "text"},
            "keywords": {"type": "text"},
            "mesh_terms_conditions": {"type": "text"},
            "mesh_terms_interventions": {"type": "text"},
        }
    }
}


def create_index():
    """Create Elasticsearch index with mappings if it doesn't exist."""
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body=mapping)
        logger.info(f"Created index {index_name}")
    else:
        logger.info(f"Index {index_name} already exists")


def get_total_trials():
    """Get the total number of trials in the studies table."""
    conn = psycopg2.connect(**pg_conn_params, connect_timeout=300)  # type: ignore
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM ctgov.studies")
    total = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    return total


class PostgreSQLStreamingFetcher:
    """
    Optimized PostgreSQL data fetcher using server-side cursor.
    This approach is more memory-efficient and allows for larger batch sizes.
    """
    
    def __init__(self, batch_size: int = 2000):
        self.batch_size = batch_size
        self.conn = None
        self.cursor = None
        
    def __enter__(self):
        # Use a named cursor for server-side cursor (doesn't load all data into memory)
        self.conn = psycopg2.connect(**pg_conn_params, connect_timeout=300)  # type: ignore
        self.cursor = self.conn.cursor(
            name='fetch_trials_cursor',
            cursor_factory=psycopg2.extras.RealDictCursor
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
    
    def fetch_batches(self):
        """
        Generator that yields batches of trials.
        
        Uses materialized view for optimal performance.
        The view pre-aggregates all data, so this query is very fast.
        
        Note: Run create_materialized_view.py first to create the view.
        """
        # Query from materialized view - much faster than subqueries or JOINs
        # All aggregations are pre-computed in the materialized view
        query = """
        SELECT
            nct_id,
            title,
            official_title,
            brief_summary,
            conditions,
            keywords,
            mesh_terms_conditions,
            interventions,
            mesh_terms_interventions
        FROM
            ctgov.studies_for_es
        ORDER BY
            nct_id;
        """
        
        self.cursor.execute(query)
        
        while True:
            # Fetch batch_size rows at a time
            rows = self.cursor.fetchmany(self.batch_size)
            if not rows:
                break
            yield rows


def transform_to_documents(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Transform PostgreSQL rows (as dicts) into Elasticsearch documents.
    RealDictCursor already returns rows as dicts, so this is simpler.
    """
    documents = []
    for row in rows:
        doc = dict(row)
        
        # Handle NULL values for array fields
        for field in [
            "conditions",
            "interventions",
            "keywords",
            "mesh_terms_conditions",
            "mesh_terms_interventions",
        ]:
            if doc.get(field) is None:
                doc[field] = []
        
        # Handle NULL brief_summary
        if doc.get("brief_summary") is None:
            doc["brief_summary"] = ""
            
        documents.append(doc)
    return documents


def generate_es_actions(documents: List[Dict[str, Any]]):
    """Generator for Elasticsearch bulk actions."""
    for doc in documents:
        yield {
            "_op_type": "index",
            "_index": index_name,
            "_id": doc["nct_id"],
            "_source": doc,
        }


def index_batch_streaming(documents: List[Dict[str, Any]]) -> Tuple[int, int]:
    """
    Bulk index using streaming_bulk for better memory efficiency.
    Returns (success_count, error_count).
    """
    success_count = 0
    error_count = 0
    
    # streaming_bulk is more memory efficient than bulk
    for ok, response in streaming_bulk(
        es,
        generate_es_actions(documents),
        chunk_size=1000,  # ES bulk request size
        raise_on_error=False,
    ):
        if ok:
            success_count += 1
        else:
            error_count += 1
            logger.error(f"Failed to index document: {response}")
    
    return success_count, error_count


def index_batch_standard(documents: List[Dict[str, Any]]) -> Tuple[int, int]:
    """Standard bulk indexing (alternative method)."""
    actions = list(generate_es_actions(documents))
    try:
        success, failed = bulk(es, actions, raise_on_error=False)
        if failed:
            logger.error(f"Failed to index {len(failed)} documents")  # type: ignore
        return success, len(failed) if failed else 0  # type: ignore
    except Exception as e:
        logger.error(f"Bulk indexing error: {e}")
        return 0, len(documents)


def main():
    """Main indexing process with optimized approach."""
    # Create index
    create_index()

    # Get total number of trials
    total_trials = get_total_trials()
    logger.info(f"Total trials to process: {total_trials}")

    # Process in batches using streaming cursor
    batch_size = 5000  # Large batch size for materialized view (pre-aggregated, very fast)
    count = 0
    time_start = time.time()
    
    logger.info(f"Starting data migration with batch size: {batch_size}")
    
    with PostgreSQLStreamingFetcher(batch_size=batch_size) as fetcher:
        for batch_num, rows in enumerate(fetcher.fetch_batches(), start=1):
            logger.info(f"Processing batch {batch_num}: {len(rows)} records")
            
            # Transform to documents
            documents = transform_to_documents(rows)
            
            # Index to Elasticsearch using streaming approach
            success, errors = index_batch_streaming(documents)
            
            count += len(documents)
            elapsed_hours = (time.time() - time_start) / 3600
            records_per_second = count / (time.time() - time_start)
            
            logger.info(
                f"Batch {batch_num}: {success} successful, {errors} errors | "
                f"Total: {count}/{total_trials} ({count*100/total_trials:.1f}%) | "
                f"Speed: {records_per_second:.1f} rec/s | "
                f"Elapsed: {elapsed_hours:.2f}h"
            )
            
            # Estimate remaining time
            if count > 0:
                estimated_total_time = (time.time() - time_start) * total_trials / count
                estimated_remaining = (estimated_total_time - (time.time() - time_start)) / 3600
                logger.info(f"Estimated time remaining: {estimated_remaining:.2f}h")

    logger.info(f"Indexing complete! Processed {count} trials in {(time.time() - time_start) / 3600:.2f} hours")


if __name__ == "__main__":
    main()
