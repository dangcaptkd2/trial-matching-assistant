from __future__ import annotations

from typing import Dict, List

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
            "brief_title",
            "official_title",
            "brief_summary",
            "detailed_description",
            "eligibility_criteria",
            "drug_name",
            "drug_keywords",
            "general_keywords",
            "primary_outcome",
            "condition",
            # "gender",
            # "minimum_age",
            # "maximum_age",
            # "intervention_type",
        ]

    def _prepare_search_text(self, patient_profile: str) -> str:
        text = (patient_profile or "").strip()
        return " ".join(text.split())

    async def search_trials(
        self,
        patient_profile: str,
        top_k: int = 20,
    ) -> List[Dict]:
        """
        Search for clinical trials matching patient profile.

        Args:
            patient_profile: Patient profile text to search
            top_k: Number of results to return

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
                "type": "cross_fields",
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

        # Run sync ES call in a thread to keep async API symmetric
        import asyncio

        def _do_search():
            return self.es.client.search(index=self.index_name, body=body)

        resp = await asyncio.get_running_loop().run_in_executor(None, _do_search)

        hits = resp.get("hits", {}).get("hits", [])
        results: List[Dict] = []
        for h in hits:
            results.append(
                {
                    "id": h.get("_id"),
                    "distance": float(h.get("_score", 0.0)),
                }
            )

        return results

    async def search_trials_by_text(
        self,
        patient_profile: str,
        top_k: int = 20,
    ) -> List[Dict]:
        """
        Search for clinical trials matching patient profile using only the 'text' field.

        This method searches only in the combined 'text' field which contains
        title, summary, description, and criteria from CTnlp ClinicalTrial.

        Args:
            patient_profile: Patient profile text to search
            top_k: Number of results to return

        Returns:
            List of trial results with id and distance (score)
        """

        query_text = self._prepare_search_text(patient_profile)
        # query_text = patient_profile
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

        # Run sync ES call in a thread to keep async API symmetric
        import asyncio

        def _do_search():
            return self.es.client.search(index=self.index_name, body=body)

        resp = await asyncio.get_running_loop().run_in_executor(None, _do_search)

        hits = resp.get("hits", {}).get("hits", [])
        results: List[Dict] = []
        for h in hits:
            results.append(
                {
                    "id": h.get("_id"),
                    "distance": float(h.get("_score", 0.0)),
                }
            )

        return results

    def format_results(self, results: List[Dict]) -> List[Dict]:
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

    async def search_and_format_by_text(
        self,
        patient_profile: str,
        top_k: int = 20,
    ) -> List[Dict]:
        """
        Search for clinical trials using only 'text' field and format results.

        Args:
            patient_profile: Patient profile text to search
            top_k: Number of results to return

        Returns:
            List of formatted trial results with id and score
        """
        results = await self.search_trials_by_text(patient_profile, top_k)
        return self.format_results(results)

    async def search_with_full_documents(
        self,
        patient_profile: str,
        top_k: int = 20,
    ) -> List[Dict]:
        """
        Search for clinical trials using 'text' field and return full document data.

        Args:
            patient_profile: Patient profile text to search
            top_k: Number of results to return

        Returns:
            List of trial results with id, score, and full document source
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

        # Run sync ES call in a thread to keep async API symmetric
        import asyncio

        def _do_search():
            return self.es.client.search(index=self.index_name, body=body)

        resp = await asyncio.get_running_loop().run_in_executor(None, _do_search)

        hits = resp.get("hits", {}).get("hits", [])
        results: List[Dict] = []
        for h in hits:
            source = h.get("_source", {})
            results.append(
                {
                    "id": h.get("_id"),
                    "score": float(h.get("_score", 0.0)),
                    "source": source,  # Full document data
                }
            )

        return results


def demo():
    import asyncio
    from src.config.settings import settings

    patient_profile = "Patient is a 45-year-old man with a history of anaplastic astrocytoma of the spine complicated by severe lower extremity weakness and urinary retention s/p Foley catheter, high-dose steroids, hypertension, and chronic pain. The tumor is located in the T-L spine, unresectable anaplastic astrocytoma s/p radiation. Complicated by progressive lower extremity weakness and urinary retention. Patient initially presented with RLE weakness where his right knee gave out with difficulty walking and right anterior thigh numbness. MRI showed a spinal cord conus mass which was biopsied and found to be anaplastic astrocytoma. Therapy included field radiation t10-l1 followed by 11 cycles of temozolomide 7 days on and 7 days off. This was followed by CPT-11 Weekly x4 with Avastin Q2 weeks/ 2 weeks rest and repeat cycle."

    async def run_demo():
        print("=== Testing Elasticsearch Trial Search (BM25) ===")
        index_name = settings.es_index_name
        searcher = ElasticsearchTrialSearcher(index_name=index_name)

        results = await searcher.search_and_format_by_text(patient_profile, top_k=5)
        print(f"Found {len(results)} trials")
        for i, trial in enumerate(results[:5], 1):
            print(f"{i}. {trial['id']} (Score: {trial['score']:.3f})")

    asyncio.run(run_demo())


if __name__ == "__main__":
    demo()
