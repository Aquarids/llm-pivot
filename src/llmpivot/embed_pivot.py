# embed_pivot.py
from typing import List, Union

from .embed.base_embed import BaseEmbed
from .config import EmbedConfig, ModelType
from .embed.api_embed import ApiEmbed
from .embed.local_embed import LocalEmbed
from .helper.logger import Logger


class EmbedPivot:

    def __init__(self, config: EmbedConfig):
        self.config = config
        self.logger = Logger(
            "EmbedPivot",
            config.log_env,
            enabled=config.enable_log,
            level=config.log_level,
        )
        self.embed: BaseEmbed = self._init_embed()

    def _init_embed(self) -> BaseEmbed:
        self.logger.info(f"Initializing EmbedPivot with model_type: {self.config.model_type}")
        self.logger.info(f"Model ID: {self.config.model_id}")

        if self.config.model_type == ModelType.ONLINE:
            if not self.config.api_key:
                self.logger.error("API key is required for online model")
                raise ValueError("api_key is required for online model")
            self.logger.info(f"Creating ApiEmbed instance, base_url: {self.config.base_url}")
            return ApiEmbed(self.config, self.logger)

        elif self.config.model_type == ModelType.LOCAL:
            self.logger.info(f"Creating LocalEmbed instance, base_url: {self.config.base_url}")
            return LocalEmbed(self.config, self.logger)

        else:
            self.logger.error(f"Unknown model_type: {self.config.model_type}")
            raise ValueError(f"Unknown model_type: {self.config.model_type}")

    def embedding(
        self,
        input: Union[str, List[str]],
        **kwargs,
    ) -> Union[List[float], List[List[float]]]:
        count = 1 if isinstance(input, str) else len(input)
        self.logger.debug(f"Embedding called with {count} text(s)")
        self.logger.debug(f"Text content: {input}")
        return self.embed.embedding(input, **kwargs)

    def cleanup(self):
        self.logger.info("Cleaning up EmbedPivot resources")
        self.embed.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def __repr__(self):
        return f"EmbedPivot(model_type={self.config.model_type}, model_id={self.config.model_id})"
