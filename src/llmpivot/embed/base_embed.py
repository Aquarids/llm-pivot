# base_embed.py
import time
from abc import ABC, abstractmethod
from typing import List, Optional

from ..config import EmbedConfig
from ..helper import Logger


class BaseEmbed(ABC):

    def __init__(self, config: EmbedConfig):
        self.config = config
        self.logger = Logger(
            name=self.__class__.__name__,
            env=config.log_env,
            enabled=config.enable_log,
            level=config.log_level,
        )

    @abstractmethod
    def _embedding_impl(
        self,
        input: List[str],
        **kwargs,
    ) -> List[List[float]]:
        pass

    def batch_embedding(
        self,
        input: List[str] | str,
        **kwargs,
    ) -> List[List[float]]:
        if isinstance(input, str):
            input = [input]

        params = {**self.config.embed_default_params, **kwargs}

        last_exc: Optional[Exception] = None
        for attempt in range(self.config.max_retries + 1):
            try:
                if attempt > 0:
                    self.logger.warning(f"Retry attempt {attempt} for embedding")
                result = self._embedding_impl(input, **params)
                self.logger.info(f"Embedding success: input_count={len(input)}")
                return result
            except Exception as e:
                last_exc = e
                self.logger.error(f"Embedding attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries:
                    time.sleep(1.0 * (attempt + 1))

        raise RuntimeError(f"Embedding failed after {self.config.max_retries + 1} attempts") from last_exc

    def embedding(
        self,
        input: str | List[str],
        **kwargs,
    ) -> List[float] | List[List[float]]:
        if isinstance(input, list):
            return self.batch_embedding(input, **kwargs)
        return self.batch_embedding([input], **kwargs)[0]

    def cleanup(self):
        pass
