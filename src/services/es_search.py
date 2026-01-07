"""Elasticsearch trial searcher"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

from src.config.settings import settings
from src.dependency.myelasticsearch import ElasticsearchService


class ElasticsearchTrialSearcher:
    """
    Clinical trial searcher using Elasticsearch (BM25).

    Supports full text search over key clinical trial fields
    per TREC 2023 guidance.
    """

    def __init__(self, index_name: str):
        self.index_name = index_name
        self.es = ElasticsearchService()

        # Fields to search (aligned with ES mappings)
        # self.search_fields = [
        #     "brief_title^3",
        #     "official_title^2",
        #     "brief_summary^3",
        #     "detailed_description",
        #     "eligibility_criteria^2",
        #     "drug_name",
        #     "drug_keywords",
        #     "general_keywords",
        #     "primary_outcome",
        #     "condition",
        # ]
        self.search_fields = [
            "title^3",
            "official_title^2",
            "brief_summary^3",
            "conditions^2",
            "interventions",
            "keywords",
            "mesh_terms_conditions",
            "mesh_terms_interventions",
        ]

    def _prepare_search_text(self, patient_profile: str) -> str:
        text = (patient_profile or "").strip()
        return " ".join(text.split())

    async def search_trials(
        self,
        patient_profile: str,
        top_k: int = 20,
        return_fields: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Search for clinical trials matching patient profile.

        Args:
            patient_profile: Patient profile text to search
            top_k: Number of results to return
            return_fields: Optional list of field names to return from Elasticsearch

        Returns:
            List of trial results with id and distance (score)
        """

        query_text = self._prepare_search_text(patient_profile)
        if not query_text:
            return []

        # Build base text search query
        query = {
            "multi_match": {
                "query": query_text,
                "fields": self.search_fields,
                "type": "best_fields",
                "operator": "or",
                # "minimum_should_match": "30%",
            }
        }

        # Note: Age filtering has been removed. The query will return all trials
        # matching the text search regardless of age constraints.

        body = {
            "size": top_k,
            "query": query,
        }

        # Add _source filter if return_fields is specified
        if return_fields is not None:
            body["_source"] = return_fields

        # Run sync ES call in a thread to keep async API symmetric
        def _do_search():
            return self.es.client.search(index=self.index_name, body=body)

        resp = await asyncio.get_running_loop().run_in_executor(None, _do_search)

        hits = resp.get("hits", {}).get("hits", [])
        results: List[Dict] = []
        for h in hits:
            result = {
                "id": h.get("_id"),
                "distance": float(h.get("_score", 0.0)),
            }
            # Add source fields if they exist
            if "_source" in h:
                result.update(h["_source"])
            results.append(result)

        return results

    async def get_trials_by_text(
        self,
        patient_profile: str,
        top_k: int = 20,
        return_fields: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Search for clinical trials matching patient profile using only the 'text' field.

        This method searches only in the combined 'text' field which contains
        title, summary, description, and criteria.

        Args:
            patient_profile: Patient profile text to search
            top_k: Number of results to return
            return_fields: Optional list of field names to return

        Returns:
            List of trial results with id, distance (score), and optionally source fields
        """

        query_text = self._prepare_search_text(patient_profile)
        if not query_text:
            return []

        # Build simple match query on 'text' field only
        query = {
            "match": {
                "text": {
                    "query": query_text,
                    "operator": "or",
                }
            }
        }

        body = {
            "size": top_k,
            "query": query,
        }

        if return_fields is not None:
            body["_source"] = return_fields

        def _do_search():
            return self.es.client.search(index=self.index_name, body=body)

        resp = await asyncio.get_running_loop().run_in_executor(None, _do_search)

        hits = resp.get("hits", {}).get("hits", [])
        results: List[Dict] = []
        for h in hits:
            result = {
                "id": h.get("_id"),
                "distance": float(h.get("_score", 0.0)),
            }
            if "_source" in h:
                result.update(h["_source"])
            results.append(result)

        return results

    def format_results(self, results: List[Dict]) -> List[Dict]:
        """Format search results for display"""
        formatted = []
        for r in results:
            formatted.append(
                {
                    "id": r.get("id"),
                    "score": r.get("distance", 0.0),
                }
            )
        return formatted

    async def search_and_format(
        self,
        patient_profile: str,
        top_k: int = 20,
    ) -> List[Dict]:
        """
        Search for clinical trials and format results.

        Args:
            patient_profile: Patient profile text to search
            top_k: Number of results to return

        Returns:
            List of formatted trial results with id and score
        """
        results = await self.search_trials(patient_profile, top_k)
        return self.format_results(results)


def demo():
    """Demo function for ElasticsearchTrialSearcher"""
    patient_profile = "Patient is a 45-year-old man with a history of anaplastic astrocytoma of the spine complicated by severe lower extremity weakness and urinary retention s/p Foley catheter, high-dose steroids, hypertension, and chronic pain. The tumor is located in the T-L spine, unresectable anaplastic astrocytoma s/p radiation. Complicated by progressive lower extremity weakness and urinary retention. Patient initially presented with RLE weakness where his right knee gave out with difficulty walking and right anterior thigh numbness. MRI showed a spinal cord conus mass which was biopsied and found to be anaplastic astrocytoma. Therapy included field radiation t10-l1 followed by 11 cycles of temozolomide 7 days on and 7 days off. This was followed by CPT-11 Weekly x4 with Avastin Q2 weeks/ 2 weeks rest and repeat cycle."

    async def run_demo():
        print("=== Testing Elasticsearch Trial Search (BM25) ===")
        index_name = settings.es_index_name
        searcher = ElasticsearchTrialSearcher(index_name=index_name)

        results = await searcher.search_trials(patient_profile, top_k=5, return_fields=["nct_id", "brief_title"])
        print(results)

    asyncio.run(run_demo())


if __name__ == "__main__":
    demo()
