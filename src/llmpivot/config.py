from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class ModelType(str, Enum):
    ONLINE = "online"
    LOCAL = "local"

class LLMApiType(str, Enum):
    CHAT_COMPLETIONS = "chat_completions"
    RESPONSES = "responses"

class PivotConfig(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    
    model_type: ModelType = Field(default=ModelType.ONLINE)
    api_type: LLMApiType = Field(default=LLMApiType.CHAT_COMPLETIONS)
    
    model_id: str = Field(default="gpt-4")
    api_key: Optional[str] = Field(default=None)
    base_url: Optional[str] = Field(default=None)
    
    timeout: int = Field(default=60, gt=0)
    max_retries: int = Field(default=1, ge=0)
    
    llm_default_params: Dict[str, Any] = Field(
        default_factory=dict
    )
    
    log_env: str = Field(default="local")
    enable_log: bool = Field(default=True)
    log_level: Optional[str] = Field(default=None)

class RerankConfig(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    model_type: ModelType = Field(default=ModelType.ONLINE)

    model_id: str = Field(default="bce-reranker-base_v1")
    api_key: Optional[str] = Field(default=None)
    base_url: Optional[str] = Field(default=None)

    timeout: int = Field(default=60, gt=0)
    max_retries: int = Field(default=1, ge=0)

    rerank_default_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "top_n": 5,
            "normalize": True,
        }
    )

    log_env: str = Field(default="local")
    enable_log: bool = Field(default=True)
    log_level: Optional[str] = Field(default=None)

class EmbedConfig(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    model_type: ModelType = Field(default=ModelType.ONLINE)

    model_id: str = Field(default="Qwen3-Embedding-0.6B")
    api_key: Optional[str] = Field(default=None)
    base_url: Optional[str] = Field(default=None)

    timeout: int = Field(default=60, gt=0)
    max_retries: int = Field(default=1, ge=0)

    embed_default_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "encoding_format": "float",
        }
    )

    log_env: str = Field(default="local")
    enable_log: bool = Field(default=True)
    log_level: Optional[str] = Field(default=None)
