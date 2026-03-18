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
    
    def _merge_params(self, **kwargs) -> Dict[str, Any]:
        params = self.config.llm_default_params.copy()
        params.update(kwargs)
        return params
    
    def _generate_impl(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        merged_params = self._merge_params(**kwargs)
        
        options = {
            "temperature": merged_params.pop("temperature", 0.7),
        }
        
        if "max_tokens" in merged_params:
            options["num_predict"] = merged_params.pop("max_tokens")
        
        ollama_option_keys = [
            "top_p", "top_k", "repeat_penalty", "repeat_last_n",
            "num_ctx", "num_batch", "num_gpu", "seed"
        ]
        for key in ollama_option_keys:
            if key in merged_params:
                options[key] = merged_params.pop(key)
        
        response = self.helper.chat(
            model=self.config.model_id,
            messages=messages,
            tools=tools,
            options=options,
            **merged_params
        )
        
        message = response['message']
        
        usage = None
        if 'eval_count' in response:
            usage = {
                "prompt_tokens": response.get('prompt_eval_count', 0),
                "completion_tokens": response.get('eval_count', 0),
                "total_tokens": response.get('prompt_eval_count', 0) + response.get('eval_count', 0),
            }
        
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
        
        return result
    
    def _embedding_impl(
        self,
        text: Union[str, List[str]],
        **kwargs
    ) -> Union[List[float], List[List[float]]]:
        merged_params = self._merge_params(**kwargs)
        
        model = merged_params.pop("model", self.config.model_id or "nomic-embed-text")
        
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        embeddings = []
        for txt in texts:
            response = self.helper.embeddings(
                model=model,
                prompt=txt,
                **merged_params
            )
            embeddings.append(response['embedding'])
        
        return embeddings[0] if is_single else embeddings
    
    def _perplexity_impl(
        self,
        text: str,
        **kwargs
    ) -> float:
        merged_params = self._merge_params(**kwargs)
        
        options = {"num_predict": merged_params.pop("max_tokens", 1)}
        
        response = self.helper.generate(
            model=self.config.model_id,
            prompt=text,
            options=options,
            **merged_params
        )
        
        if 'prompt_eval_count' in response:
            token_count = response['prompt_eval_count']
            log_likelihood = -token_count * 0.5
            ppl = math.exp(-log_likelihood / max(token_count, 1))
            return ppl
        
        raise ValueError("Unable to calculate perplexity")
    
    def _generate_stream_impl(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Iterator[str]:
        merged_params = self._merge_params(**kwargs)
        
        options = {
            "temperature": merged_params.pop("temperature", 0.7),
        }
        
        if "max_tokens" in merged_params:
            options["num_predict"] = merged_params.pop("max_tokens")
        
        ollama_option_keys = [
            "top_p", "top_k", "repeat_penalty", "repeat_last_n",
            "num_ctx", "num_batch", "num_gpu", "seed"
        ]
        for key in ollama_option_keys:
            if key in merged_params:
                options[key] = merged_params.pop(key)
        
        stream = self.helper.chat(
            model=self.config.model_id,
            messages=messages,
            tools=tools,
            stream=True,
            options=options,
            **merged_params
        )
        
        for chunk in stream:
            if chunk['message'].get('content'):
                yield chunk['message']['content']
    
    def cleanup(self):
        self.logger.info("Cleaning up LocalLLM resources")
        self.helper.stop()
