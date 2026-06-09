# local_rerank.py
from typing import List, Dict, Any

from .base_rerank import BaseRerank
from ..config import RerankConfig
from ..helper.infinity_helper import InfinityHelper


class LocalRerank(BaseRerank):

    def __init__(self, config: RerankConfig):
        super().__init__(config)
        self.infinity = InfinityHelper(
            logger=self.logger,
            host=config.base_url,
            model_id=config.model_id,
        )
        self.infinity.start()

    def _rerank_impl(
        self,
        query: str,
        documents: List[str],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        return self.infinity.rerank(
            model=self.config.model_id,
            query=query,
            documents=documents,
            **kwargs,
        )

    def _similarity_impl(
        self,
        source_sentence: str,
        sentences: List[str],
        **kwargs,
    ) -> List[float]:
        return self.infinity.similarity(
            model=self.config.model_id,
            source_sentence=source_sentence,
            sentences=sentences,
            **kwargs,
        )

    def cleanup(self):
        self.infinity.stop()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass
