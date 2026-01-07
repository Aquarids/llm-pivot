from typing import List, Dict, Any, Optional, Union, Iterator
from openai import OpenAI

from .base_llm import BaseLLM
from ..helper import Logger
from ..config import PivotConfig

class ApiLLM(BaseLLM):
    
    def __init__(self, config: PivotConfig, logger: Logger):
        super().__init__(config, logger)
        self.logger.info(f"Initializing ApiLLM with model: {config.model_id}")

        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
        )
    
    def _generate_impl(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        params = {
            "model": self.config.model_id,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
        }
        
        if self.config.max_tokens:
            params["max_tokens"] = kwargs.get("max_tokens", self.config.max_tokens)
        
        if tools:
            params["tools"] = tools
            if "tool_choice" in kwargs:
                params["tool_choice"] = kwargs["tool_choice"]
        
        self.logger.debug(f"Calling OpenAI API with params: model={params['model']}, temperature={params['temperature']}")
        response = self.client.chat.completions.create(**params)
        
        choice = response.choices[0]
        message = choice.message
        
        self.logger.info(f"API response received, tokens: {response.usage.total_tokens}")
        
        return {
            "content": message.content,
            "role": message.role,
            "tool_calls": message.tool_calls,
            "finish_reason": choice.finish_reason,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            "model": response.model,
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
        kwargs.setdefault("tool_choice", "auto")
        response = self._generate_impl(messages, tools=tools, **kwargs)
        
        if not response["tool_calls"]:
            self.logger.warning("No tool calls in response")
            return []
        
        result = []
        for tc in response["tool_calls"]:
            result.append({
                "id": tc.id,
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            })
        
        self.logger.info(f"Function calls: {len(result)} tool(s) called")
        return result
    
    def _embedding_impl(
        self,
        text: Union[str, List[str]],
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        model = kwargs.get("model", self.config.model_id)
        if not model:
            raise ValueError("Invalid Model ID")
        
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        self.logger.debug(f"Calling embedding API with model: {model}, texts: {len(texts)}")
        response = self.client.embeddings.create(
            model=model,
            input=texts,
        )
        
        embeddings = [item.embedding for item in response.data]
        self.logger.info(f"Embeddings generated: {len(embeddings)} vector(s), dim={len(embeddings[0])}")
        
        return embeddings[0] if is_single else embeddings
    
    def _perplexity_impl(
        self,
        text: str,
        **kwargs
    ) -> float:
        self.logger.debug("Calculating perplexity using logprobs")
        response = self.client.chat.completions.create(
            model=self.config.model_id,
            messages=[{"role": "user", "content": text}],
            max_tokens=1,
            logprobs=True,
            **kwargs
        )
        
        if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
            logprobs = response.choices[0].logprobs.content
            if logprobs:
                log_likelihood = sum(token.logprob for token in logprobs)
                token_count = len(logprobs)
                ppl = 2 ** (-log_likelihood / token_count)
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
        params = {
            "model": self.config.model_id,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "stream": True,
        }
        
        if tools:
            params["tools"] = tools
        
        stream = self.client.chat.completions.create(**params)
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
