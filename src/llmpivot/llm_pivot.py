from typing import List, Dict, Any, Optional, Union, Iterator
from .config import PivotConfig, ModelType
from .llm import ApiLLM, LocalLLM, BaseLLM
from .helper import Logger

class LLMPivot:
    
    def __init__(self, config: PivotConfig):
        self.config = config
        self.logger = Logger(
            "LLMPivot", 
            config.log_env,
            enabled=config.enable_log,
            level=config.log_level
        )
        self.llm: BaseLLM = self._init_llm()
    
    def _init_llm(self) -> BaseLLM:
        self.logger.info(f"Initializing LLMPivot with model_type: {self.config.model_type}")
        self.logger.info(f"Model ID: {self.config.model_id}")
        
        if self.config.model_type == ModelType.ONLINE:
            if not self.config.api_key:
                self.logger.error("API key is required for online model")
                raise ValueError("api_key is required for online model")
            
            self.logger.info(f"Creating ApiLLM instance, base_url: {self.config.base_url}")
            return ApiLLM(self.config, self.logger)
        
        elif self.config.model_type == ModelType.LOCAL:
            self.logger.info(f"Creating LocalLLM instance, ollama_host: {self.config.base_url}")
            return LocalLLM(self.config, self.logger)
        
        else:
            self.logger.error(f"Unknown model_type: {self.config.model_type}")
            raise ValueError(f"Unknown model_type: {self.config.model_type}")
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        self.logger.debug(f"Generate called with {len(messages)} messages")
        self.logger.debug(f"Messages content: {messages}")
        if tools:
            self.logger.debug(f"Tools provided: {len(tools)} tool(s)")
        return self.llm.generate(messages, tools, **kwargs)
    
    def dialogue(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs
    ) -> str:
        self.logger.debug(f"Dialogue called with {len(messages)} messages (stream={stream}, params={kwargs})")
        self.logger.debug(f"Messages content: {messages}")
        return self.llm.dialogue(messages, stream, **kwargs)

    def dialogue_with_usage(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        self.logger.debug(f"Dialogue with usage called with {len(messages)} messages (stream={stream}, params={kwargs})")
        self.logger.debug(f"Messages content: {messages}")
        return self.llm.dialogue_with_usage(messages, stream, **kwargs)
    
    def call_function(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        self.logger.debug(f"Call function with {len(tools)} tools")
        self.logger.debug(f"Messages content: {messages}")
        self.logger.debug(f"Tools content: {tools}")
        return self.llm.call_function(messages, tools, **kwargs)
    
    def embedding(
        self,
        text: Union[str, List[str]],
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        text_count = 1 if isinstance(text, str) else len(text)
        self.logger.debug(f"Embedding called with {text_count} text(s)")
        self.logger.debug(f"Text content: {text}")
        return self.llm.embedding(text, **kwargs)
    
    def perplexity(
        self,
        text: str,
        **kwargs
    ) -> float:
        self.logger.debug("Perplexity called")
        self.logger.debug(f"Text content: {text}")
        return self.llm.perplexity(text, **kwargs)
    
    def stream_generate(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Iterator[str]:
        self.logger.debug("Stream generate called")
        self.logger.debug(f"Messages content: {messages}")
        if tools:
            self.logger.debug(f"Tools provided: {len(tools)} tool(s)")
        return self.llm.stream_generate(messages, tools, **kwargs)
    
    def cleanup(self):
        self.logger.info("Cleaning up LLMPivot resources")
        self.llm.cleanup()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
    
    def __repr__(self):
        return f"LLMPivot(model_type={self.config.model_type}, model_id={self.config.model_id})"
