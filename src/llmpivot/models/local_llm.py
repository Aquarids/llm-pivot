from typing import List, Dict, Any, Optional, Union, Iterator
import math

from ..helper import Logger
from ..helper import OllamaHelper
from ..config import PivotConfig
from .base_llm import BaseLLM

class LocalLLM(BaseLLM):
    
    def __init__(self, config: PivotConfig, logger: Logger):
        super().__init__(config, logger)
        self.logger.info(f"Initializing LocalLLM with model: {config.model_id}")
        self.helper = OllamaHelper(logger=logger, host=config.base_url)
        
        self.helper.start()
        self.helper.ensure_model(config.model_id)
    
    def _generate_impl(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        options = {
            "temperature": kwargs.get("temperature", self.config.temperature),
        }
        
        if self.config.max_tokens:
            options["num_predict"] = kwargs.get("max_tokens", self.config.max_tokens)
        
        self.logger.debug(f"Calling Ollama with model: {self.config.model_id}")
        response = self.helper.chat(
            model=self.config.model_id,
            messages=messages,
            tools=tools,
            options=options,
        )
        
        message = response['message']
        
        usage = None
        if 'eval_count' in response:
            usage = {
                "prompt_tokens": response.get('prompt_eval_count', 0),
                "completion_tokens": response.get('eval_count', 0),
                "total_tokens": response.get('prompt_eval_count', 0) + response.get('eval_count', 0),
            }
            self.logger.info(f"Ollama response received, tokens: {usage['total_tokens']}")
        
        return {
            "content": message.get('content'),
            "role": message['role'],
            "tool_calls": message.get('tool_calls'),
            "finish_reason": 'stop',
            "usage": usage,
            "model": response.get('model', self.config.model_id),
        }
    
    def _dialogue_impl(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        response = self._generate_impl(messages, tools=None, **kwargs)
        return response["content"] or ""
    
    def _call_function_impl(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        response = self._generate_impl(messages, tools=tools, **kwargs)
        
        if not response["tool_calls"]:
            self.logger.warning("No tool calls in response")
            return []
        
        result = []
        for tc in response["tool_calls"]:
            if hasattr(tc, 'function'):
                result.append({
                    "id": getattr(tc, 'id', f"call_{len(result)}"),
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                })
            else:
                result.append({
                    "id": tc.get('id', f"call_{len(result)}"),
                    "name": tc['function']['name'],
                    "arguments": tc['function']['arguments'],
                })
        
        self.logger.info(f"Function calls: {len(result)} tool(s) called")
        return result
    
    def _embedding_impl(
        self,
        text: Union[str, List[str]],
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        model = kwargs.get("model", self.config.model_id or "nomic-embed-text")
        
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        self.logger.debug(f"Generating embeddings for {len(texts)} text(s)")
        embeddings = []
        for txt in texts:
            response = self.helper.embeddings(model=model, prompt=txt)
            embeddings.append(response['embedding'])
        
        self.logger.info(f"Embeddings generated: {len(embeddings)} vector(s), dim={len(embeddings[0])}")
        
        return embeddings[0] if is_single else embeddings
    
    def _perplexity_impl(
        self,
        text: str,
        **kwargs
    ) -> float:
        self.logger.debug("Calculating perplexity using Ollama generate")
        response = self.helper.generate(
            model=self.config.model_id,
            prompt=text,
            options={"num_predict": 1}
        )
        
        if 'prompt_eval_count' in response:
            token_count = response['prompt_eval_count']
            log_likelihood = -token_count * 0.5
            ppl = math.exp(-log_likelihood / max(token_count, 1))
            self.logger.info(f"Perplexity calculated: {ppl:.2f}")
            return ppl
        
        self.logger.error("Unable to calculate perplexity from response")
        raise ValueError("Unable to calculate perplexity")
    
    def stream_generate(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Iterator[str]:
        self.logger.debug("Starting stream generation")
        options = {
            "temperature": kwargs.get("temperature", self.config.temperature),
        }
        
        stream = self.helper.chat(
            model=self.config.model_id,
            messages=messages,
            tools=tools,
            stream=True,
            options=options,
        )
        
        for chunk in stream:
            if chunk['message'].get('content'):
                yield chunk['message']['content']
    
    def cleanup(self):
        self.logger.info("Cleaning up LocalLLM resources")
        self.helper.stop()
