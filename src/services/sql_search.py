"""PostgreSQL trial searcher for AACT database."""

from __future__ import annotations

import asyncio
from typing import Dict, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

from src.config.settings import settings


def get_trial_by_id(trial_id: str) -> Optional[Dict]:
    """
    Retrieve a clinical trial by NCT ID from PostgreSQL AACT database.

    Args:
        trial_id: NCT ID of the trial (e.g., "NCT01234567")

    Returns:
        Dictionary containing trial information, or None if not found
    """
    # Build connection parameters
    if not all([settings.postgres_host, settings.postgres_database, settings.postgres_user]):
        raise ValueError(
            "PostgreSQL configuration is incomplete. "
            "Please set POSTGRES_HOST, POSTGRES_DATABASE, and POSTGRES_USER in your .env file."
        )

    conn_params = {
        "host": settings.postgres_host,
        "port": settings.postgres_port,
        "database": settings.postgres_database,
        "user": settings.postgres_user,
        "client_encoding": "utf8",
        "connect_timeout": 10,          # fail fast if DB is unreachable
        "options": "-c statement_timeout=25000",  # 25s query timeout
    }

    if settings.postgres_password:
        conn_params["password"] = settings.postgres_password

    # Use CTEs to pre-aggregate each table independently per nct_id.
    # This avoids the combinatorial row explosion caused by joining many large
    # tables before aggregation (which produced billions of intermediate rows).
    query = """
        WITH
        cond AS (
            SELECT nct_id,
                   array_agg(DISTINCT name) AS conditions,
                   STRING_AGG(DISTINCT name, E'\n') AS condition_name
            FROM {schema}.conditions WHERE nct_id = %s GROUP BY nct_id
        ),
        interv AS (
            SELECT nct_id,
                   array_agg(DISTINCT name) AS interventions,
                   STRING_AGG(DISTINCT intervention_type || ': ' || name, E'\n') AS intervention_name
            FROM {schema}.interventions WHERE nct_id = %s GROUP BY nct_id
        ),
        kw AS (
            SELECT nct_id, array_agg(DISTINCT name) AS keywords
            FROM {schema}.keywords WHERE nct_id = %s GROUP BY nct_id
        ),
        bc AS (
            SELECT nct_id, array_agg(DISTINCT mesh_term) AS mesh_terms_conditions
            FROM {schema}.browse_conditions WHERE nct_id = %s GROUP BY nct_id
        ),
        bi AS (
            SELECT nct_id, array_agg(DISTINCT mesh_term) AS mesh_terms_interventions
            FROM {schema}.browse_interventions WHERE nct_id = %s GROUP BY nct_id
        ),
        fac AS (
            SELECT nct_id,
                   COUNT(DISTINCT CONCAT_WS(', ', name, city, state, country)) AS total_sites,
                   STRING_AGG(DISTINCT CONCAT_WS(', ', name, city, state, country), E'\n') AS sites
            FROM {schema}.facilities WHERE nct_id = %s GROUP BY nct_id
        ),
        rc AS (
            SELECT nct_id,
                   name || ' ' || COALESCE(phone,'') || ' ' || COALESCE(email,'') AS contact
            FROM {schema}.result_contacts WHERE nct_id = %s LIMIT 1
        )
        SELECT
            s.nct_id,
            s.brief_title            AS title,
            s.official_title,
            s.overall_status,
            s.phase,
            s.start_date,
            s.completion_date,
            s.overall_status         AS status,
            bs.description           AS brief_summary,
            e.criteria               AS eligibility_criteria,
            rc.contact,
            COALESCE(cond.conditions, '{{}}')               AS conditions,
            COALESCE(cond.condition_name, '')               AS condition_name,
            COALESCE(interv.interventions, '{{}}')          AS interventions,
            COALESCE(interv.intervention_name, '')          AS intervention_name,
            COALESCE(kw.keywords, '{{}}')                   AS keywords,
            COALESCE(bc.mesh_terms_conditions, '{{}}')      AS mesh_terms_conditions,
            COALESCE(bi.mesh_terms_interventions, '{{}}')   AS mesh_terms_interventions,
            COALESCE(fac.total_sites, 0)                    AS total_sites,
            COALESCE(fac.sites, '')                         AS sites
        FROM {schema}.studies s
        LEFT JOIN {schema}.brief_summaries bs ON bs.nct_id = s.nct_id
        LEFT JOIN {schema}.eligibilities   e  ON e.nct_id  = s.nct_id
        LEFT JOIN cond   ON cond.nct_id   = s.nct_id
        LEFT JOIN interv ON interv.nct_id = s.nct_id
        LEFT JOIN kw     ON kw.nct_id     = s.nct_id
        LEFT JOIN bc     ON bc.nct_id     = s.nct_id
        LEFT JOIN bi     ON bi.nct_id     = s.nct_id
        LEFT JOIN fac    ON fac.nct_id    = s.nct_id
        LEFT JOIN rc     ON rc.nct_id     = s.nct_id
        WHERE s.nct_id = %s;
    """.format(schema=settings.postgres_schema)

    try:
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # 8 parameters: one per CTE WHERE clause + final WHERE
        cursor.execute(query, (trial_id,) * 8)
        result = cursor.fetchone()

        cursor.close()
        conn.close()

        return dict(result) if result else None

    except psycopg2.Error as e:
        raise RuntimeError(f"Database error while fetching trial {trial_id}: {str(e)}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error while fetching trial {trial_id}: {str(e)}") from e


def get_trials_by_ids(trial_ids: list[str]) -> list[Dict]:
    """
    Retrieve multiple clinical trials by NCT IDs from PostgreSQL AACT database.

    Args:
        trial_ids: List of NCT IDs (e.g., ["NCT01234567", "NCT07654321"])

    Returns:
        List of dictionaries containing trial information
    """
    if not trial_ids:
        return []

    # Build connection parameters
    if not all([settings.postgres_host, settings.postgres_database, settings.postgres_user]):
        raise ValueError(
            "PostgreSQL configuration is incomplete. "
            "Please set POSTGRES_HOST, POSTGRES_DATABASE, and POSTGRES_USER in your .env file."
        )

    conn_params = {
        "host": settings.postgres_host,
        "port": settings.postgres_port,
        "database": settings.postgres_database,
        "user": settings.postgres_user,
        "client_encoding": "utf8",
        "connect_timeout": 10,          # fail fast if DB is unreachable
        "options": "-c statement_timeout=25000",  # 25s query timeout
    }

    if settings.postgres_password:
        conn_params["password"] = settings.postgres_password

    # Use CTEs to pre-aggregate each table independently per nct_id.
    # Avoids the combinatorial row explosion from joining many large tables.
    query = """
        WITH
        cond AS (
            SELECT nct_id,
                   array_agg(DISTINCT name) AS conditions,
                   STRING_AGG(DISTINCT name, E'\n') AS condition_name
            FROM {schema}.conditions WHERE nct_id = ANY(%s) GROUP BY nct_id
        ),
        interv AS (
            SELECT nct_id,
                   array_agg(DISTINCT name) AS interventions,
                   STRING_AGG(DISTINCT intervention_type || ': ' || name, E'\n') AS intervention_name
            FROM {schema}.interventions WHERE nct_id = ANY(%s) GROUP BY nct_id
        ),
        kw AS (
            SELECT nct_id, array_agg(DISTINCT name) AS keywords
            FROM {schema}.keywords WHERE nct_id = ANY(%s) GROUP BY nct_id
        ),
        bc AS (
            SELECT nct_id, array_agg(DISTINCT mesh_term) AS mesh_terms_conditions
            FROM {schema}.browse_conditions WHERE nct_id = ANY(%s) GROUP BY nct_id
        ),
        bi AS (
            SELECT nct_id, array_agg(DISTINCT mesh_term) AS mesh_terms_interventions
            FROM {schema}.browse_interventions WHERE nct_id = ANY(%s) GROUP BY nct_id
        ),
        fac AS (
            SELECT nct_id,
                   COUNT(DISTINCT CONCAT_WS(', ', name, city, state, country)) AS total_sites,
                   STRING_AGG(DISTINCT CONCAT_WS(', ', name, city, state, country), E'\n') AS sites
            FROM {schema}.facilities WHERE nct_id = ANY(%s) GROUP BY nct_id
        ),
        rc AS (
            SELECT DISTINCT ON (nct_id) nct_id,
                   name || ' ' || COALESCE(phone,'') || ' ' || COALESCE(email,'') AS contact
            FROM {schema}.result_contacts WHERE nct_id = ANY(%s) ORDER BY nct_id
        )
        SELECT
            s.nct_id,
            s.brief_title            AS title,
            s.official_title,
            s.overall_status,
            s.phase,
            s.start_date,
            s.completion_date,
            s.overall_status         AS status,
            bs.description           AS brief_summary,
            e.criteria               AS eligibility_criteria,
            rc.contact,
            COALESCE(cond.conditions, '{{}}')               AS conditions,
            COALESCE(cond.condition_name, '')               AS condition_name,
            COALESCE(interv.interventions, '{{}}')          AS interventions,
            COALESCE(interv.intervention_name, '')          AS intervention_name,
            COALESCE(kw.keywords, '{{}}')                   AS keywords,
            COALESCE(bc.mesh_terms_conditions, '{{}}')      AS mesh_terms_conditions,
            COALESCE(bi.mesh_terms_interventions, '{{}}')   AS mesh_terms_interventions,
            COALESCE(fac.total_sites, 0)                    AS total_sites,
            COALESCE(fac.sites, '')                         AS sites
        FROM {schema}.studies s
        LEFT JOIN {schema}.brief_summaries bs ON bs.nct_id = s.nct_id
        LEFT JOIN {schema}.eligibilities   e  ON e.nct_id  = s.nct_id
        LEFT JOIN cond   ON cond.nct_id   = s.nct_id
        LEFT JOIN interv ON interv.nct_id = s.nct_id
        LEFT JOIN kw     ON kw.nct_id     = s.nct_id
        LEFT JOIN bc     ON bc.nct_id     = s.nct_id
        LEFT JOIN bi     ON bi.nct_id     = s.nct_id
        LEFT JOIN fac    ON fac.nct_id    = s.nct_id
        LEFT JOIN rc     ON rc.nct_id     = s.nct_id
        WHERE s.nct_id = ANY(%s);
    """.format(schema=settings.postgres_schema)

    try:
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # 8 parameters: one per CTE WHERE clause + final WHERE
        cursor.execute(query, (trial_ids,) * 8)
        results = cursor.fetchall()

        cursor.close()
        conn.close()

        return [dict(result) for result in results] if results else []

    except psycopg2.Error as e:
        raise RuntimeError(f"Database error while fetching trials {trial_ids}: {str(e)}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error while fetching trials {trial_ids}: {str(e)}") from e


async def get_trials_by_ids_async(trial_ids: list[str]) -> list[Dict]:
    """
    Async wrapper for get_trials_by_ids.

    Args:
        trial_ids: List of NCT IDs (e.g., ["NCT01234567", "NCT07654321"])

    Returns:
        List of dictionaries containing trial information
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_trials_by_ids, trial_ids)


async def get_trial_by_id_async(trial_id: str) -> Optional[Dict]:
    """
    Async wrapper for get_trial_by_id.

    Args:
        trial_id: NCT ID of the trial (e.g., "NCT01234567")

    Returns:
        Dictionary containing trial information, or None if not found
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, get_trial_by_id, trial_id)


def group_locations(sites_str: str, max_locations: int = 4) -> list[str]:
    """
    Parse and group location strings from PostgreSQL sites field.

    Args:
        sites_str: String of sites in format "city, state, country\ncity2, state2, country2..."
        max_locations: Maximum number of unique locations to return (default: 4)

    Returns:
        List of unique location strings, limited to max_locations
    """
    if not sites_str or sites_str == "N/A":
        return []

    # Split by newline and clean up
    locations = [loc.strip() for loc in sites_str.split("\n") if loc.strip()]

    # Remove duplicates while preserving order
    seen = set()
    unique_locations = []
    for loc in locations:
        # Normalize for comparison (case-insensitive)
        loc_normalized = loc.lower()
        if loc_normalized not in seen:
            seen.add(loc_normalized)
            unique_locations.append(loc)

    # Return first max_locations
    return unique_locations[:max_locations]


if __name__ == "__main__":
    TRIAL_ID = "NCT01783236"
    print(get_trial_by_id(TRIAL_ID))
