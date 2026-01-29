from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class ModelType(str, Enum):
    ONLINE = "online"
    LOCAL = "local"

class PivotConfig(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    
    model_type: ModelType = Field(default=ModelType.ONLINE)
    
    model_id: str = Field(default="gpt-4")
    api_key: Optional[str] = Field(default=None)
    base_url: Optional[str] = Field(default=None)
    
    timeout: int = Field(default=60, gt=0)
    max_retries: int = Field(default=1, ge=0)
    
    llm_default_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "temperature": 0.7,
        }
    )
    
    log_env: str = Field(default="local")
    enable_log: bool = Field(default=True)
    log_level: Optional[str] = Field(default=None)
