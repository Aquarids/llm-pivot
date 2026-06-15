from typing import List, Dict, Any, Optional, Union, Iterator
from openai import OpenAI

from .base_llm import BaseLLM
from ..helper import Logger
from ..config import PivotConfig

class ApiLLM(BaseLLM):
    
    def __init__(self, config: PivotConfig, logger: Logger):
        super().__init__(config, logger)
        self.logger.info(f"Initializing ApiLLM with model: {config.model_id}")
        self.last_stream_usage = None

        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
        )
    
    def _merge_params(self, **kwargs) -> Dict[str, Any]:
        params = self.config.llm_default_params.copy()
        params.update(kwargs)
        return params

    def _usage_to_dict(self, usage: Any) -> Dict[str, int]:
        if isinstance(usage, dict):
            return {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }

        return {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }
    
    def _generate_impl(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        merged_params = self._merge_params(**kwargs)
        
        params = {
            "model": self.config.model_id,
            "messages": messages,
            "temperature": merged_params.pop("temperature", 0.7),
        }
        
        if "max_tokens" in merged_params:
            params["max_tokens"] = merged_params.pop("max_tokens")
        
        if tools:
            params["tools"] = tools
            if "tool_choice" in merged_params:
                params["tool_choice"] = merged_params.pop("tool_choice")
        
        params.update(merged_params)
        
        response = self.client.chat.completions.create(**params)
        
        choice = response.choices[0]
        message = choice.message
        
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
            return []
        
        result = []
        for tc in response["tool_calls"]:
            result.append({
                "id": tc.id,
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            })
        
        return result
    
    def _embedding_impl(
        self,
        text: Union[str, List[str]],
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        merged_params = self._merge_params(**kwargs)
        
        model = merged_params.pop("model", self.config.model_id)
        if not model:
            raise ValueError("Invalid Model ID")
        
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        params = {
            "model": model,
            "input": texts,
        }
        params.update(merged_params)
        
        response = self.client.embeddings.create(**params)
        
        embeddings = [item.embedding for item in response.data]
        
        return embeddings[0] if is_single else embeddings
    
    def _perplexity_impl(
        self,
        text: str,
        **kwargs
    ) -> float:
        merged_params = self._merge_params(**kwargs)
        
        params = {
            "model": self.config.model_id,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": merged_params.pop("max_tokens", 1),
            "logprobs": True,
        }
        params.update(merged_params)
        
        response = self.client.chat.completions.create(**params)
        
        if hasattr(response.choices[0], 'logprobs') and response.choices[0].logprobs:
            logprobs = response.choices[0].logprobs.content
            if logprobs:
                log_likelihood = sum(token.logprob for token in logprobs)
                token_count = len(logprobs)
                ppl = 2 ** (-log_likelihood / token_count)
                return ppl
        
        raise ValueError("Unable to calculate perplexity")
    
    def _generate_stream_impl(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Iterator[str]:
        merged_params = self._merge_params(**kwargs)
        
        params = {
            "model": self.config.model_id,
            "messages": messages,
            "temperature": merged_params.pop("temperature", 0.7),
            "stream": True,
        }
        stream_options = merged_params.pop("stream_options", {})
        params["stream_options"] = {"include_usage": True, **stream_options}
        
        if "max_tokens" in merged_params:
            params["max_tokens"] = merged_params.pop("max_tokens")
        
        if tools:
            params["tools"] = tools
            if "tool_choice" in merged_params:
                params["tool_choice"] = merged_params.pop("tool_choice")
        
        params.update(merged_params)
        
        self.last_stream_usage = None
        stream = self.client.chat.completions.create(**params)
        
        for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                self.last_stream_usage = self._usage_to_dict(usage)

            if (chunk.choices 
                and len(chunk.choices) > 0 
                and chunk.choices[0].delta.content):
                yield chunk.choices[0].delta.content
