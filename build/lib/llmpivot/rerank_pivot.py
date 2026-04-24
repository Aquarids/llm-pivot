# rerank_pivot.py
from typing import List, Dict, Any

from .rerank.base_rerank import BaseRerank
from .config import RerankConfig, ModelType
from .rerank.api_rerank import ApiRerank
from .rerank.local_rerank import LocalRerank
from .helper.logger import Logger


class RerankPivot:

    def __init__(self, config: RerankConfig):
        self.config = config
        self.logger = Logger(
            "RerankPivot",
            config.log_env,
            enabled=config.enable_log,
            level=config.log_level,
        )
        self.rerank: BaseRerank = self._init_rerank()

    def _init_rerank(self) -> BaseRerank:
        self.logger.info(f"Initializing RerankPivot with model_type: {self.config.model_type}")
        self.logger.info(f"Model ID: {self.config.model_id}")

        if self.config.model_type == ModelType.ONLINE:
            if not self.config.api_key:
                self.logger.error("API key is required for online model")
                raise ValueError("api_key is required for online model")
            self.logger.info(f"Creating ApiRerank instance, base_url: {self.config.base_url}")
            return ApiRerank(self.config, self.logger)

        elif self.config.model_type == ModelType.LOCAL:
            self.logger.info(f"Creating LocalRerank instance, base_url: {self.config.base_url}")
            return LocalRerank(self.config, self.logger)

        else:
            self.logger.error(f"Unknown model_type: {self.config.model_type}")
            raise ValueError(f"Unknown model_type: {self.config.model_type}")

    def rerank_documents(
        self,
        query: str,
        documents: List[str],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        self.logger.debug(f"Rerank called with query={query!r}, docs={len(documents)}")
        return self.rerank.rerank(query, documents, **kwargs)

    def similarity(
        self,
        source_sentence: str,
        sentences: List[str],
        **kwargs,
    ) -> List[float]:
        self.logger.debug(f"Similarity called with {len(sentences)} sentence(s)")
        return self.rerank.similarity(source_sentence, sentences, **kwargs)

    def cleanup(self):
        self.logger.info("Cleaning up RerankPivot resources")
        self.rerank.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def __repr__(self):
        return f"RerankPivot(model_type={self.config.model_type}, model_id={self.config.model_id})"
