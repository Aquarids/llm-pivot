from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Iterator
import time

from ..helper import Logger
from ..config import PivotConfig


class BaseLLM(ABC):
    
    def __init__(self, config: PivotConfig, logger: Logger):
        self.config = config
        self.logger = logger
    
    def _retry_wrapper(self, func, *args, **kwargs):
        last_error = None
        for attempt in range(self.config.max_retries + 1):
            try:
                if attempt > 0:
                    self.logger.warning(f"Retry attempt {attempt}/{self.config.max_retries}")
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                self.logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.config.max_retries:
                    sleep_time = 2 ** attempt
                    self.logger.info(f"Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    self.logger.error(f"All {self.config.max_retries + 1} attempts failed")
                    self.logger.log_exception(last_error)
                    raise last_error
    
    @abstractmethod
    def _generate_impl(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        pass
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        return self._retry_wrapper(self._generate_impl, messages, tools, **kwargs)
    
    @abstractmethod
    def _dialogue_impl(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        pass
    
    @abstractmethod
    def _generate_stream_impl(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Iterator[str]:
        pass
    
    def dialogue(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs
    ) -> str:
        if stream:
            try:
                chunks = []
                for chunk in self._generate_stream_impl(messages, tools=None, **kwargs):
                    chunks.append(chunk)
                result = "".join(chunks)
                return result
            except Exception as e:
                self.logger.log_exception(e)
                return self._retry_wrapper(self._dialogue_impl, messages, **kwargs)
        else:
            return self._retry_wrapper(self._dialogue_impl, messages, **kwargs)

    def dialogue_with_usage(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        if stream:
            try:
                chunks = []
                for chunk in self._generate_stream_impl(messages, tools=None, **kwargs):
                    chunks.append(chunk)
                return {
                    "content": "".join(chunks),
                    "usage": getattr(self, "last_stream_usage", None),
                }
            except Exception as e:
                self.logger.log_exception(e)

        response = self._retry_wrapper(self._generate_impl, messages, None, **kwargs)
        return {
            "content": response["content"] or "",
            "usage": response.get("usage"),
        }
    
    def stream_generate(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Iterator[str]:
        return self._generate_stream_impl(messages, tools, **kwargs)
    
    @abstractmethod
    def _call_function_impl(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        pass
    
    def call_function(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        return self._retry_wrapper(self._call_function_impl, messages, tools, **kwargs)
    
    @abstractmethod
    def _embedding_impl(
        self,
        text: Union[str, List[str]],
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        pass
    
    def embedding(
        self,
        text: Union[str, List[str]],
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        return self._embedding_impl(text, kwargs)
    
    @abstractmethod
    def _perplexity_impl(
        self,
        text: str,
        **kwargs
    ) -> float:
        pass
    
    def perplexity(
        self,
        text: str,
        **kwargs
    ) -> float:
        return self._perplexity_impl(text, kwargs)
    
    def cleanup(self):
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
