# api_embed.py
from typing import List
from openai import OpenAI

from .base_embed import BaseEmbed
from ..config import EmbedConfig


class ApiEmbed(BaseEmbed):

    def __init__(self, config: EmbedConfig):
        super().__init__(config)
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=0,
        )

    def _embedding_impl(
        self,
        input: List[str],
        **kwargs,
    ) -> List[List[float]]:
        response = self.client.embeddings.create(
            model=self.config.model_id,
            input=input,
            **kwargs,
        )
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]
