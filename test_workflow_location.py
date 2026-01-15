"""
Test workflow location search integration.

This script tests the complete workflow with location-based queries to ensure
the intent classification correctly extracts location information and the search
node properly filters trials by location.
"""

import asyncio

from src.core.graph import create_workflow


async def test_workflow_location_search():
    """Test location-based search through the workflow."""

    print("=" * 70)
    print("WORKFLOW LOCATION SEARCH INTEGRATION TEST")
    print("=" * 70)

    # Create workflow
    app = create_workflow()

    # Test cases
    test_cases = [
        {
            "name": "Lung cancer in Vietnam",
            "query": "Find trials for lung cancer in Vietnam",
            "expected_location": "Vietnam",
            "expected_patient_info": "lung cancer",
        },
        {
            "name": "Diabetes in Ho Chi Minh City",
            "query": "I have diabetes and I live in Ho Chi Minh City",
            "expected_location": "Ho Chi Minh City",
            "expected_patient_info": "diabetes",
        },
        {
            "name": "Breast cancer at Memorial Hospital",
            "query": "Breast cancer trials at Memorial Hospital",
            "expected_location": "Memorial Hospital",
            "expected_patient_info": "breast cancer",
        },
        {
            "name": "No location specified",
            "query": "Find trials for lung cancer",
            "expected_location": None,
            "expected_patient_info": "lung cancer",
        },
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'=' * 70}")
        print(f"Test {i}: {test['name']}")
        print(f"{'=' * 70}")
        print(f"Query: {test['query']}")

        # Run workflow
        config = {"configurable": {"thread_id": f"test-location-{i}"}}
        initial_state = {
            "user_input": test["query"],
            "messages": [],
            "top_k": 5,
        }

        try:
            result = await app.ainvoke(initial_state, config=config)

            # Check extracted fields
            intent_type = result.get("intent_type")
            patient_info = result.get("patient_info")
            location_info = result.get("location_info")
            search_results = result.get("search_results", [])
            final_answer = result.get("final_answer", "")

            print(f"\n✓ Intent: {intent_type}")
            print(f"✓ Patient Info: {patient_info}")
            print(f"✓ Location Info: {location_info}")
            print(f"✓ Search Results: {len(search_results)} trials found")

            # Verify expectations
            if patient_info != test["expected_patient_info"]:
                print(f"⚠️  Expected patient_info: {test['expected_patient_info']}")

            if location_info != test["expected_location"]:
                print(f"⚠️  Expected location_info: {test['expected_location']}")

            # Show sample results
            if search_results:
                print("\nTop results:")
                for j, trial in enumerate(search_results[:2], 1):
                    nct_id = trial.get("nct_id", "N/A")
                    title = trial.get("title", "N/A")
                    print(f"  {j}. {nct_id}: {title[:60]}...")

            # Show final answer preview
            if final_answer:
                print("\nFinal Answer (preview):")
                print(f"  {final_answer[:150]}...")

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'=' * 70}")
    print("ALL TESTS COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    asyncio.run(test_workflow_location_search())
