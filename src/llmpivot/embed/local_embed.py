# local_embed.py
from typing import List

from .base_embed import BaseEmbed
from ..config import EmbedConfig
from ..helper.infinity_helper import InfinityHelper


class LocalEmbed(BaseEmbed):

    def __init__(self, config: EmbedConfig):
        super().__init__(config)
        self.infinity = InfinityHelper(
            model_id=config.model_id,
            base_url=config.base_url,
        )
        self.infinity.start()

    def _embedding_impl(
        self,
        input: List[str],
        **kwargs,
    ) -> List[List[float]]:
        return self.infinity.embeddings(
            model=self.config.model_id,
            input=input,
            **kwargs,
        )

    def __del__(self):
        try:
            self.infinity.stop()
        except Exception:
            pass
