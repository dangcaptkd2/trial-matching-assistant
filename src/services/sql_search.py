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
    }

    if settings.postgres_password:
        conn_params["password"] = settings.postgres_password

    # Modified query to filter by specific trial_id
    query = """
        SELECT
            s.nct_id,
            s.brief_title as title,
            s.official_title,
            s.overall_status,
            s.phase,
            s.start_date,
            s.completion_date,
            array_agg(DISTINCT c.name) AS conditions,
            array_agg(DISTINCT i.name) AS interventions,
            array_agg(DISTINCT k.name) AS keywords,
            array_agg(DISTINCT bc.mesh_term) AS mesh_terms_conditions,
            array_agg(DISTINCT bi.mesh_term) AS mesh_terms_interventions,
            bs.description AS brief_summary,
            STRING_AGG(DISTINCT i.intervention_type || ': ' || i.name, E'\n') AS intervention_name,
            STRING_AGG(DISTINCT c.name, E'\n') AS condition_name,
            r.name || ' ' || r.phone || ' ' || r.email AS contact,
            COUNT(DISTINCT CONCAT_WS(', ', f.city, f.state, f.country)) AS total_sites,
            STRING_AGG(DISTINCT CONCAT_WS(', ', f.city, f.state, f.country), E'\n') AS sites,
            s.overall_status AS status,
            e.criteria AS eligibility_criteria
        FROM
            {schema}.studies s
        LEFT JOIN
            {schema}.conditions c ON s.nct_id = c.nct_id
        LEFT JOIN
            {schema}.interventions i ON s.nct_id = i.nct_id
        LEFT JOIN
            {schema}.keywords k ON s.nct_id = k.nct_id
        LEFT JOIN
            {schema}.browse_conditions bc ON s.nct_id = bc.nct_id
        LEFT JOIN
            {schema}.browse_interventions bi ON s.nct_id = bi.nct_id
        LEFT JOIN
            {schema}.brief_summaries bs ON s.nct_id = bs.nct_id
        LEFT JOIN
            {schema}.result_contacts AS r ON s.nct_id = r.nct_id
        LEFT JOIN
            {schema}.facilities AS f ON s.nct_id = f.nct_id
        LEFT JOIN
            {schema}.eligibilities e ON s.nct_id = e.nct_id
        WHERE
            s.nct_id = %s
        GROUP BY
            s.nct_id, s.brief_title, s.official_title, s.overall_status, s.phase, s.start_date, s.completion_date, bs.description, r.name, r.phone, r.email, e.criteria;
    """.format(schema=settings.postgres_schema)

    try:
        # Connect to database
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Execute query
        cursor.execute(query, (trial_id,))
        result = cursor.fetchone()

        # Close connection
        cursor.close()
        conn.close()

        # Convert result to dictionary if found
        if result:
            return dict(result)
        return None

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
    }

    if settings.postgres_password:
        conn_params["password"] = settings.postgres_password

    # Modified query to filter by multiple trial_ids using IN clause
    query = """
        SELECT
            s.nct_id,
            s.brief_title as title,
            s.official_title,
            s.overall_status,
            s.phase,
            s.start_date,
            s.completion_date,
            array_agg(DISTINCT c.name) AS conditions,
            array_agg(DISTINCT i.name) AS interventions,
            array_agg(DISTINCT k.name) AS keywords,
            array_agg(DISTINCT bc.mesh_term) AS mesh_terms_conditions,
            array_agg(DISTINCT bi.mesh_term) AS mesh_terms_interventions,
            bs.description AS brief_summary,
            STRING_AGG(DISTINCT i.intervention_type || ': ' || i.name, E'\n') AS intervention_name,
            STRING_AGG(DISTINCT c.name, E'\n') AS condition_name,
            r.name || ' ' || r.phone || ' ' || r.email AS contact,
            COUNT(DISTINCT CONCAT_WS(', ',f.name, f.city, f.state, f.country)) AS total_sites,
            STRING_AGG(DISTINCT CONCAT_WS(', ', f.name, f.city, f.state, f.country), E'\n') AS sites,
            s.overall_status AS status,
            e.criteria AS eligibility_criteria
        FROM
            {schema}.studies s
        LEFT JOIN
            {schema}.conditions c ON s.nct_id = c.nct_id
        LEFT JOIN
            {schema}.interventions i ON s.nct_id = i.nct_id
        LEFT JOIN
            {schema}.keywords k ON s.nct_id = k.nct_id
        LEFT JOIN
            {schema}.browse_conditions bc ON s.nct_id = bc.nct_id
        LEFT JOIN
            {schema}.browse_interventions bi ON s.nct_id = bi.nct_id
        LEFT JOIN
            {schema}.brief_summaries bs ON s.nct_id = bs.nct_id
        LEFT JOIN
            {schema}.result_contacts AS r ON s.nct_id = r.nct_id
        LEFT JOIN
            {schema}.facilities AS f ON s.nct_id = f.nct_id
        LEFT JOIN
            {schema}.eligibilities e ON s.nct_id = e.nct_id
        WHERE
            s.nct_id = ANY(%s)
        GROUP BY
            s.nct_id, s.brief_title, s.official_title, s.overall_status, s.phase, s.start_date, s.completion_date, bs.description, r.name, r.phone, r.email, e.criteria;
    """.format(schema=settings.postgres_schema)

    try:
        # Connect to database
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Execute query with list of trial IDs
        cursor.execute(query, (trial_ids,))
        results = cursor.fetchall()

        # Close connection
        cursor.close()
        conn.close()

        # Convert results to list of dictionaries
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
