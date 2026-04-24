# base_rerank.py
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from ..config import RerankConfig
from ..helper import Logger


class BaseRerank(ABC):

    def __init__(self, config: RerankConfig):
        self.config = config
        self.logger = Logger(
            name=self.__class__.__name__,
            env=config.log_env,
            enable=config.enable_log,
            level=config.log_level,
        )

    @abstractmethod
    def _rerank_impl(
        self,
        query: str,
        documents: List[str],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def _similarity_impl(
        self,
        source_sentence: str,
        sentences: List[str],
        **kwargs,
    ) -> List[float]:
        pass

    def rerank(
        self,
        query: str,
        documents: List[str],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        params = {**self.config.rerank_default_params, **kwargs}

        last_exc: Optional[Exception] = None
        for attempt in range(self.config.max_retries + 1):
            try:
                if attempt > 0:
                    self.logger.warning(f"Retry attempt {attempt} for rerank")
                result = self._rerank_impl(query, documents, **params)
                self.logger.info(f"Rerank success: docs={len(documents)}")
                return result
            except Exception as e:
                last_exc = e
                self.logger.error(f"Rerank attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries:
                    time.sleep(1.0 * (attempt + 1))

        raise RuntimeError(f"Rerank failed after {self.config.max_retries + 1} attempts") from last_exc

    def similarity(
        self,
        source_sentence: str,
        sentences: List[str],
        **kwargs,
    ) -> List[float]:
        params = {**self.config.rerank_default_params, **kwargs}

        last_exc: Optional[Exception] = None
        for attempt in range(self.config.max_retries + 1):
            try:
                if attempt > 0:
                    self.logger.warning(f"Retry attempt {attempt} for similarity")
                result = self._similarity_impl(source_sentence, sentences, **params)
                self.logger.info(f"Similarity success: sentences={len(sentences)}")
                return result
            except Exception as e:
                last_exc = e
                self.logger.error(f"Similarity attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries:
                    time.sleep(1.0 * (attempt + 1))

        raise RuntimeError(f"Similarity failed after {self.config.max_retries + 1} attempts") from last_exc
