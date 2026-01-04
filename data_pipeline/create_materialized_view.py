"""Create and manage materialized view for optimized Elasticsearch indexing.

This script creates a materialized view that pre-aggregates data from multiple
tables in the AACT database. This significantly speeds up the Elasticsearch
indexing process by avoiding complex JOINs during the data export.

Usage:
    python data_pipeline/create_materialized_view.py              # Create view
    python data_pipeline/create_materialized_view.py --refresh    # Refresh existing view
    python data_pipeline/create_materialized_view.py --drop       # Drop view
"""

import argparse
import sys
import time

import psycopg2
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

MATERIALIZED_VIEW_NAME = "ctgov.studies_for_es"


def create_materialized_view():
    """
    Create materialized view with pre-aggregated data from multiple tables.
    
    This view combines data from:
    - studies: Main study information
    - conditions: Study conditions (aggregated)
    - interventions: Study interventions (aggregated)
    - keywords: Study keywords (aggregated)
    - browse_conditions: MeSH terms for conditions (aggregated)
    - browse_interventions: MeSH terms for interventions (aggregated)
    - brief_summaries: Study summaries
    """
    logger.info(f"Creating materialized view: {MATERIALIZED_VIEW_NAME}")
    
    create_view_sql = f"""
    CREATE MATERIALIZED VIEW {MATERIALIZED_VIEW_NAME} AS
    SELECT
        s.nct_id,
        s.brief_title as title,
        s.official_title,
        (SELECT bs.description 
         FROM ctgov.brief_summaries bs 
         WHERE bs.nct_id = s.nct_id 
         LIMIT 1) AS brief_summary,
        (SELECT array_agg(DISTINCT c.name) 
         FROM ctgov.conditions c 
         WHERE c.nct_id = s.nct_id) AS conditions,
        (SELECT array_agg(DISTINCT k.name) 
         FROM ctgov.keywords k 
         WHERE k.nct_id = s.nct_id) AS keywords,
        (SELECT array_agg(DISTINCT bc.mesh_term) 
         FROM ctgov.browse_conditions bc 
         WHERE bc.nct_id = s.nct_id) AS mesh_terms_conditions,
        (SELECT array_agg(DISTINCT i.name) 
         FROM ctgov.interventions i 
         WHERE i.nct_id = s.nct_id) AS interventions,
        (SELECT array_agg(DISTINCT bi.mesh_term) 
         FROM ctgov.browse_interventions bi 
         WHERE bi.nct_id = s.nct_id) AS mesh_terms_interventions
    FROM
        ctgov.studies s;
    """
    
    create_index_sql = f"""
    CREATE INDEX idx_studies_for_es_nct 
    ON {MATERIALIZED_VIEW_NAME}(nct_id);
    """
    
    conn = None
    try:
        conn = psycopg2.connect(**pg_conn_params, connect_timeout=300)  # type: ignore
        cursor = conn.cursor()
        
        # Check if view already exists
        cursor.execute(f"""
            SELECT EXISTS (
                SELECT 1 
                FROM pg_matviews 
                WHERE schemaname = 'ctgov' 
                AND matviewname = 'studies_for_es'
            );
        """)
        exists = cursor.fetchone()[0]
        
        if exists:
            logger.warning(f"Materialized view {MATERIALIZED_VIEW_NAME} already exists")
            logger.info("Use --refresh to refresh it or --drop to drop it first")
            cursor.close()
            conn.close()
            return False
        
        # Create materialized view
        logger.info("Creating materialized view (this may take several minutes)...")
        start_time = time.time()
        
        cursor.execute(create_view_sql)
        conn.commit()
        
        duration = time.time() - start_time
        logger.success(f"✓ Materialized view created in {duration:.1f} seconds")
        
        # Create index
        logger.info("Creating index on materialized view...")
        start_time = time.time()
        
        cursor.execute(create_index_sql)
        conn.commit()
        
        duration = time.time() - start_time
        logger.success(f"✓ Index created in {duration:.1f} seconds")
        
        # Get statistics
        cursor.execute(f"SELECT COUNT(*) FROM {MATERIALIZED_VIEW_NAME}")
        count = cursor.fetchone()[0]
        logger.info(f"Materialized view contains {count:,} studies")
        
        # Get view size
        cursor.execute(f"""
            SELECT pg_size_pretty(pg_total_relation_size('{MATERIALIZED_VIEW_NAME}'));
        """)
        size = cursor.fetchone()[0]
        logger.info(f"Materialized view size: {size}")
        
        cursor.close()
        conn.close()
        
        logger.success("✅ Materialized view created successfully!")
        return True
        
    except psycopg2.Error as e:
        logger.error(f"Database error: {e}")
        if conn:
            conn.rollback()
            conn.close()
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if conn:
            conn.close()
        return False


def refresh_materialized_view():
    """
    Refresh materialized view with latest data.
    
    This should be run after importing new data into PostgreSQL.
    CONCURRENTLY option allows queries to continue during refresh.
    """
    logger.info(f"Refreshing materialized view: {MATERIALIZED_VIEW_NAME}")
    
    # Note: CONCURRENTLY requires a UNIQUE index on at least one column
    refresh_sql = f"REFRESH MATERIALIZED VIEW CONCURRENTLY {MATERIALIZED_VIEW_NAME};"
    
    conn = None
    try:
        conn = psycopg2.connect(**pg_conn_params, connect_timeout=300)  # type: ignore
        cursor = conn.cursor()
        
        # Check if view exists
        cursor.execute(f"""
            SELECT EXISTS (
                SELECT 1 
                FROM pg_matviews 
                WHERE schemaname = 'ctgov' 
                AND matviewname = 'studies_for_es'
            );
        """)
        exists = cursor.fetchone()[0]
        
        if not exists:
            logger.error(f"Materialized view {MATERIALIZED_VIEW_NAME} does not exist")
            logger.info("Run without --refresh flag to create it first")
            cursor.close()
            conn.close()
            return False
        
        # Refresh
        logger.info("Refreshing materialized view (this may take several minutes)...")
        start_time = time.time()
        
        cursor.execute(refresh_sql)
        conn.commit()
        
        duration = time.time() - start_time
        logger.success(f"✓ Materialized view refreshed in {duration / 60:.1f} minutes")
        
        # Get statistics
        cursor.execute(f"SELECT COUNT(*) FROM {MATERIALIZED_VIEW_NAME}")
        count = cursor.fetchone()[0]
        logger.info(f"Materialized view contains {count:,} studies")
        
        cursor.close()
        conn.close()
        
        logger.success("✅ Materialized view refreshed successfully!")
        return True
        
    except psycopg2.Error as e:
        logger.error(f"Database error: {e}")
        if conn:
            conn.rollback()
            conn.close()
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if conn:
            conn.close()
        return False


def drop_materialized_view():
    """Drop the materialized view."""
    logger.info(f"Dropping materialized view: {MATERIALIZED_VIEW_NAME}")
    
    drop_sql = f"DROP MATERIALIZED VIEW IF EXISTS {MATERIALIZED_VIEW_NAME} CASCADE;"
    
    conn = None
    try:
        conn = psycopg2.connect(**pg_conn_params, connect_timeout=300)  # type: ignore
        cursor = conn.cursor()
        
        cursor.execute(drop_sql)
        conn.commit()
        
        cursor.close()
        conn.close()
        
        logger.success("✓ Materialized view dropped successfully!")
        return True
        
    except psycopg2.Error as e:
        logger.error(f"Database error: {e}")
        if conn:
            conn.rollback()
            conn.close()
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if conn:
            conn.close()
        return False


def get_view_info():
    """Get information about the materialized view."""
    conn = None
    try:
        conn = psycopg2.connect(**pg_conn_params, connect_timeout=300)  # type: ignore
        cursor = conn.cursor()
        
        # Check if view exists
        cursor.execute(f"""
            SELECT EXISTS (
                SELECT 1 
                FROM pg_matviews 
                WHERE schemaname = 'ctgov' 
                AND matviewname = 'studies_for_es'
            );
        """)
        exists = cursor.fetchone()[0]
        
        if not exists:
            logger.info(f"Materialized view {MATERIALIZED_VIEW_NAME} does not exist")
            cursor.close()
            conn.close()
            return
        
        # Get view info
        logger.info(f"Materialized View: {MATERIALIZED_VIEW_NAME}")
        logger.info("-" * 60)
        
        # Row count
        cursor.execute(f"SELECT COUNT(*) FROM {MATERIALIZED_VIEW_NAME}")
        count = cursor.fetchone()[0]
        logger.info(f"Row count: {count:,}")
        
        # Size
        cursor.execute(f"""
            SELECT pg_size_pretty(pg_total_relation_size('{MATERIALIZED_VIEW_NAME}'));
        """)
        size = cursor.fetchone()[0]
        logger.info(f"Size: {size}")
        
        # Last refresh (if available)
        # Note: PostgreSQL doesn't track refresh time by default
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error getting view info: {e}")
        if conn:
            conn.close()


def main():
    """Main function to handle command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Manage materialized view for Elasticsearch indexing"
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh existing materialized view with latest data"
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop the materialized view"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show information about the materialized view"
    )
    
    args = parser.parse_args()
    
    # Handle different operations
    if args.info:
        get_view_info()
    elif args.drop:
        if drop_materialized_view():
            sys.exit(0)
        else:
            sys.exit(1)
    elif args.refresh:
        if refresh_materialized_view():
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        # Default: create view
        if create_materialized_view():
            sys.exit(0)
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()
