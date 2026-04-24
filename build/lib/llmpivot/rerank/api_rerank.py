# api_rerank.py
from typing import List, Dict, Any
import httpx

from .base_rerank import BaseRerank
from ..config import RerankConfig


class ApiRerank(BaseRerank):

    def __init__(self, config: RerankConfig):
        super().__init__(config)
        self.client = httpx.Client(
            base_url=config.base_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.api_key}",
            },
            timeout=config.timeout,
        )

    def _rerank_impl(
        self,
        query: str,
        documents: List[str],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        payload = {
            "model": self.config.model_id,
            "query": query,
            "documents": documents,
            **kwargs,
        }
        response = self.client.post("/v1/rerank", json=payload)
        response.raise_for_status()
        data = response.json()
        sorted_results = sorted(data["results"], key=lambda x: x["index"])
        return [
            {
                "index": item["index"],
                "text": item["document"]["text"],
                "relevance_score": item["relevance_score"],
            }
            for item in sorted_results
        ]

    def _similarity_impl(
        self,
        source_sentence: str,
        sentences: List[str],
        **kwargs,
    ) -> List[float]:
        payload = {
            "model": self.config.model_id,
            "inputs": {
                "source_sentence": source_sentence,
                "sentences": sentences,
            },
            **kwargs,
        }
        response = self.client.post("/v1/sentence-similarity", json=payload)
        response.raise_for_status()
        return response.json()
