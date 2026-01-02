import time

import psycopg2
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from loguru import logger

from src.config.settings import settings

# Loguru logger is used directly

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
            "overall_status": {"type": "keyword"},
            "phase": {"type": "keyword"},
            "start_date": {"type": "date"},
            "completion_date": {"type": "date"},
            "conditions": {"type": "text"},
            "interventions": {"type": "text"},
            "keywords": {"type": "text"},
            "mesh_terms_conditions": {"type": "text"},
            "mesh_terms_interventions": {"type": "text"},
            "brief_summary": {"type": "text"},
            "intervention_name": {"type": "text"},
            "condition_name": {"type": "text"},
            "contact": {"type": "text"},
            "total_sites": {"type": "integer"},
            "sites": {"type": "text"},
            "status": {"type": "keyword"},
            "eligibility_criteria": {"type": "text"},
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


def fetch_batch(last_nct_id=None, limit=500):
    """Fetch a batch of trials from PostgreSQL using keyset pagination."""
    conn = psycopg2.connect(**pg_conn_params, connect_timeout=300)  # type: ignore
    cursor = conn.cursor()
    # query = """
    # SELECT
    #     s.nct_id,
    #     s.brief_title as title,
    #     s.official_title,
    #     s.overall_status,
    #     s.phase,
    #     s.start_date,
    #     s.completion_date,
    #     array_agg(DISTINCT c.name) AS conditions,
    #     array_agg(DISTINCT i.name) AS interventions,
    #     array_agg(DISTINCT k.name) AS keywords,
    #     array_agg(DISTINCT bc.mesh_term) AS mesh_terms_conditions,
    #     array_agg(DISTINCT bi.mesh_term) AS mesh_terms_interventions,
    #     bs.description AS brief_summary,
    #     STRING_AGG(DISTINCT i.intervention_type || ': ' || i.name, E'\n') AS intervention_name,
    #     STRING_AGG(DISTINCT c.name, E'\n') AS condition_name,
    #     r.name || ' ' || r.phone || ' ' || r.email AS contact,
    #     COUNT(DISTINCT CONCAT_WS(', ', f.city, f.state, f.country)) AS total_sites,
    #     STRING_AGG(DISTINCT CONCAT_WS(', ', f.city, f.state, f.country), E'\n') AS sites,
    #     s.overall_status AS status,
    #     e.criteria AS eligibility_criteria
    # FROM
    #     ctgov.studies s
    # LEFT JOIN
    #     ctgov.conditions c ON s.nct_id = c.nct_id
    # LEFT JOIN
    #     ctgov.interventions i ON s.nct_id = i.nct_id
    # LEFT JOIN
    #     ctgov.keywords k ON s.nct_id = k.nct_id
    # LEFT JOIN
    #     ctgov.browse_conditions bc ON s.nct_id = bc.nct_id
    # LEFT JOIN
    #     ctgov.browse_interventions bi ON s.nct_id = bi.nct_id
    # LEFT JOIN
    #     ctgov.brief_summaries bs ON s.nct_id = bs.nct_id
    # LEFT JOIN
    #     ctgov.result_contacts AS r ON s.nct_id = r.nct_id
    # LEFT JOIN
    #     ctgov.facilities AS f ON s.nct_id = f.nct_id
    # LEFT JOIN
    #     ctgov.eligibilities e ON s.nct_id = e.nct_id
    # WHERE
    #     s.nct_id > %s
    # GROUP BY
    #     s.nct_id, s.brief_title, s.official_title, s.overall_status, s.phase, s.start_date, s.completion_date, bs.description, r.name, r.phone, r.email, e.criteria
    # ORDER BY
    #     s.nct_id
    # LIMIT %s;
    # """

    # query = """
    # SELECT
    #     s.nct_id,
    #     s.brief_title as title,
    #     s.official_title,
    #     s.overall_status,
    #     s.phase,
    #     s.start_date,
    #     s.completion_date,
    #     array_agg(DISTINCT c.name) AS conditions,
    #     array_agg(DISTINCT i.name) AS interventions,
    #     array_agg(DISTINCT k.name) AS keywords,
    #     array_agg(DISTINCT bc.mesh_term) AS mesh_terms_conditions,
    #     array_agg(DISTINCT bi.mesh_term) AS mesh_terms_interventions,
    #     bs.description AS brief_summary,
    #     STRING_AGG(DISTINCT i.intervention_type || ': ' || i.name, E'\n') AS intervention_name,
    #     STRING_AGG(DISTINCT c.name, E'\n') AS condition_name,
    #     r.name || ' ' || r.phone || ' ' || r.email AS contact,
    #     COUNT(DISTINCT CONCAT_WS(', ', f.city, f.state, f.country)) AS total_sites,
    #     STRING_AGG(DISTINCT CONCAT_WS(', ', f.city, f.state, f.country), E'\n') AS sites,
    #     s.overall_status AS status,
    #     e.criteria AS eligibility_criteria
    # FROM
    #     ctgov.studies s
    # LEFT JOIN
    #     ctgov.conditions c ON s.nct_id = c.nct_id
    # LEFT JOIN
    #     ctgov.interventions i ON s.nct_id = i.nct_id
    # LEFT JOIN
    #     ctgov.keywords k ON s.nct_id = k.nct_id
    # LEFT JOIN
    #     ctgov.browse_conditions bc ON s.nct_id = bc.nct_id
    # LEFT JOIN
    #     ctgov.browse_interventions bi ON s.nct_id = bi.nct_id
    # LEFT JOIN
    #     ctgov.brief_summaries bs ON s.nct_id = bs.nct_id
    # LEFT JOIN
    #     ctgov.result_contacts AS r ON s.nct_id = r.nct_id
    # LEFT JOIN
    #     ctgov.facilities AS f ON s.nct_id = f.nct_id
    # LEFT JOIN
    #     ctgov.eligibilities e ON s.nct_id = e.nct_id
    # WHERE
    #     s.nct_id > %s
    # GROUP BY
    #     s.nct_id, s.brief_title, s.official_title, s.overall_status, s.phase, s.start_date, s.completion_date, bs.description, r.name, r.phone, r.email, e.criteria
    # ORDER BY
    #     s.nct_id
    # LIMIT %s;
    # """

    query = """
    SELECT
        s.nct_id,
        s.brief_title as title,
        s.official_title,
        bs.description AS brief_summary,
        array_agg(DISTINCT c.name) AS conditions,
        array_agg(DISTINCT k.name) AS keywords,
        array_agg(DISTINCT bc.mesh_term) AS mesh_terms_conditions,
        array_agg(DISTINCT i.name) AS interventions,
        array_agg(DISTINCT bi.mesh_term) AS mesh_terms_interventions

    FROM
        ctgov.studies s
    LEFT JOIN
        ctgov.conditions c ON s.nct_id = c.nct_id
    LEFT JOIN
        ctgov.interventions i ON s.nct_id = i.nct_id
    LEFT JOIN
        ctgov.keywords k ON s.nct_id = k.nct_id
    LEFT JOIN
        ctgov.browse_conditions bc ON s.nct_id = bc.nct_id
    LEFT JOIN
        ctgov.browse_interventions bi ON s.nct_id = bi.nct_id
    LEFT JOIN
        ctgov.brief_summaries bs ON s.nct_id = bs.nct_id
    LEFT JOIN
        ctgov.result_contacts AS r ON s.nct_id = r.nct_id
    WHERE
        s.nct_id > %s
    GROUP BY
        s.nct_id, s.brief_title, s.official_title, bs.description
    ORDER BY
        s.nct_id
    LIMIT %s;
    """

    # Use an empty string or a very low value for the first batch
    cursor.execute(query, (last_nct_id or "", limit))
    rows = cursor.fetchall()
    column_names = [desc[0] for desc in cursor.description]
    cursor.close()
    conn.close()

    # Return the rows, column names, and the last nct_id for the next batch
    next_nct_id = rows[-1][0] if rows else None
    return rows, column_names, next_nct_id


def transform_to_documents(rows, column_names):
    """Transform PostgreSQL rows into JSON documents."""
    documents = []
    for row in rows:
        doc = {column_names[i]: row[i] for i in range(len(column_names))}
        # Handle NULL values for array fields
        for field in [
            "conditions",
            "interventions",
            "keywords",
            "mesh_terms_conditions",
            "mesh_terms_interventions",
            # "sites",
            # "intervention_name",
            # "condition_name",
        ]:
            if doc[field] is None:
                doc[field] = []
            # Convert string-aggregated fields to lists
            # elif field in ["sites", "intervention_name", "condition_name"]:
            #     doc[field] = doc[field].split("\n") if doc[field] else []

        # # Ensure contact field exists and handle NULL
        # if "contact" in doc and doc["contact"] is None:
        #     doc["contact"] = ""

        # # Ensure eligibility_criteria exists and handle NULL
        # if "eligibility_criteria" in doc and doc["eligibility_criteria"] is None:
        #     doc["eligibility_criteria"] = ""

        documents.append(doc)
    return documents


def index_batch(documents):
    """Bulk index a batch of documents to Elasticsearch."""
    actions = [
        {
            "_op_type": "index",
            "_index": index_name,
            "_id": doc["nct_id"],
            "_source": doc,
        }
        for doc in documents
    ]
    try:
        success, failed = bulk(es, actions, raise_on_error=False)
        logger.info(f"Successfully indexed {success} documents")
        if failed:
            logger.error(f"Failed to index {len(failed)} documents: {failed}")  # type: ignore
        return success, failed
    except Exception as e:
        logger.error(f"Bulk indexing error: {e}")
        return 0, [{"error": str(e)}]


def main():
    # Create index
    create_index()

    # Get total number of trials
    total_trials = get_total_trials()
    logger.info(f"Total trials to process: {total_trials}")

    # Process in batches
    last_nct_id = ""
    batch_size = 50000
    count = 0
    time_start = time.time()
    while True:
        logger.info(f"Fetching batch at last_nct_id {last_nct_id}")
        rows, column_names, next_nct_id = fetch_batch(last_nct_id, batch_size)
        if not rows:
            logger.info("No more data to fetch")
            break

        documents = transform_to_documents(rows, column_names)
        logger.info(f"Transformed {len(documents)} documents")

        success, failed = index_batch(documents)
        logger.info(f"Indexed batch: {success} successful, {len(failed)} failed")  # type: ignore

        last_nct_id = next_nct_id

        count += len(documents)
        logger.info(
            f"Processed {count} trials in {((time.time() - time_start) / 3600):.2f} hours"
        )

    logger.info("Indexing complete")


if __name__ == "__main__":
    main()
