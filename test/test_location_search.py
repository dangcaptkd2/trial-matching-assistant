"""
Test script for location-based search functionality - Vietnam test.

This script verifies that the location search feature works correctly
for finding clinical trials in Vietnam.
"""

import asyncio

from src.config.settings import settings
from src.services.es_search import ElasticsearchTrialSearcher


async def test_vietnam_location():
    """Test Vietnam location search."""
    searcher = ElasticsearchTrialSearcher(settings.es_index_name)

    print("=" * 70)
    print("VIETNAM LOCATION SEARCH TEST")
    print("=" * 70)

    # Test: Search for trials in Vietnam
    print("\n🇻🇳 Searching for Clinical Trials in Vietnam")
    print("-" * 70)

    try:
        results = await searcher.search_trials_by_location(
            country="Vietnam",
            top_k=20,
            return_fields=["nct_id", "title", "countries", "cities", "facility_names", "conditions"],
        )

        print(f"\n✓ Found {len(results)} trials in Vietnam\n")

        if results:
            for i, r in enumerate(results, 1):
                print(f"{i}. {r['id']}: {r.get('title', 'N/A')}")

                # Show cities
                cities = r.get("cities", [])
                if cities:
                    print(f"   📍 Cities: {', '.join(cities[:5])}")

                # Show facilities
                facilities = r.get("facility_names", [])
                if facilities:
                    print(f"   🏥 Facilities: {facilities[0]}")
                    if len(facilities) > 1:
                        print(f"      + {len(facilities) - 1} more facility(ies)")

                # Show conditions
                conditions = r.get("conditions", [])
                if conditions:
                    print(f"   🔬 Conditions: {', '.join(conditions[:3])}")

                print()
        else:
            print("⚠️  No trials found in Vietnam")
            print("\nTrying alternative search with 'Viet Nam' (with space)...")

            # Try with space
            results = await searcher.search_trials_by_location(
                country="Viet Nam", top_k=20, return_fields=["nct_id", "title", "countries", "cities", "facility_names"]
            )

            if results:
                print(f"✓ Found {len(results)} trials with 'Viet Nam'\n")
                for i, r in enumerate(results[:5], 1):
                    print(f"{i}. {r['id']}: {r.get('title', 'N/A')[:70]}...")
            else:
                print("⚠️  Still no results. Trying free-text search...")

                # Try free-text search
                results = await searcher.search_trials_by_location(
                    location_text="Vietnam", top_k=20, return_fields=["nct_id", "title", "facility_names", "sites_text"]
                )

                if results:
                    print(f"✓ Found {len(results)} trials with 'Vietnam' in location text\n")
                    for i, r in enumerate(results[:5], 1):
                        print(f"{i}. {r['id']}: {r.get('title', 'N/A')[:70]}...")
                        facilities = r.get("facility_names", [])
                        if facilities:
                            print(f"   Facilities: {facilities[:2]}")
                else:
                    print("❌ No Vietnam trials found in database")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_vietnam_location())
