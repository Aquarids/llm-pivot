from .__version__ import __version__
from .llm_pivot import LLMPivot
from .embed_pivot import EmbedPivot
from .rerank_pivot import RerankPivot
from .config import PivotConfig, EmbedConfig, RerankConfig, ModelType, LLMApiType, SIWCConfig

__all__ = [
    "__version__",
    
    "LLMPivot",
    "EmbedPivot",
    "RerankPivot",
    "PivotConfig",
    "EmbedConfig",
    "RerankConfig",
    "ModelType",
    "LLMApiType",
    "SIWCConfig",
    "SIWCAuth",
]


def __getattr__(name):
    if name == "SIWCAuth":
        from .siwc.auth import SIWCAuth
        return SIWCAuth
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
