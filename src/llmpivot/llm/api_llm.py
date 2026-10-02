import json
from typing import List, Dict, Any, Optional, Union, Iterator
from openai import OpenAI
from anthropic import Anthropic

from .base_llm import BaseLLM
from ..helper import Logger
from ..config import LLMApiType, PivotConfig

class ApiLLM(BaseLLM):

    def __init__(self, config: PivotConfig, logger: Logger):
        super().__init__(config, logger)
        self.logger.info(f"Initializing ApiLLM with model: {config.model_id}")
        self.last_stream_usage = None

        if self._is_anthropic():
            self.client = self._init_anthropic_client(config)
        else:
            self.client = OpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=config.timeout,
            )

    def _is_anthropic(self) -> bool:
        return self.config.api_type == LLMApiType.ANTHROPIC

    def _init_anthropic_client(self, config: PivotConfig) -> Anthropic:
        base_url = (config.base_url or "").rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[: -len("/v1")]
        return Anthropic(
            auth_token=config.api_key,
            base_url=base_url or None,
            timeout=config.timeout,
        )
    
    def _merge_params(self, **kwargs) -> Dict[str, Any]:
        params = self.config.llm_default_params.copy()
        params.update(kwargs)
        return params

    def _usage_to_dict(self, usage: Any) -> Dict[str, int]:
        if isinstance(usage, dict):
            return {
                "prompt_tokens": usage.get("prompt_tokens", usage.get("input_tokens", 0)),
                "completion_tokens": usage.get("completion_tokens", usage.get("output_tokens", 0)),
                "total_tokens": usage.get("total_tokens", 0),
            }

        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", getattr(usage, "input_tokens", 0)),
            "completion_tokens": getattr(usage, "completion_tokens", getattr(usage, "output_tokens", 0)),
            "total_tokens": usage.total_tokens,
        }

    def _anthropic_usage(self, usage: Any) -> Dict[str, int]:
        if isinstance(usage, dict):
            input_tokens = usage.get("input_tokens", 0) or 0
            output_tokens = usage.get("output_tokens", 0) or 0
            cache_read = usage.get("cache_read_input_tokens", 0) or 0
            cache_creation = usage.get("cache_creation_input_tokens", 0) or 0
        else:
            input_tokens = getattr(usage, "input_tokens", 0) or 0
            output_tokens = getattr(usage, "output_tokens", 0) or 0
            cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
            cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
        prompt_tokens = input_tokens + cache_read + cache_creation
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": prompt_tokens + output_tokens,
        }

    def _content_to_text(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
            return "".join(parts)
        return str(content)

    def _anthropic_messages(self, messages: List[Dict[str, Any]]):
        system_parts = []
        converted = []

        def append(role: str, content: Any):
            if converted and converted[-1]["role"] == role:
                prev = converted[-1]["content"]
                if isinstance(prev, str) and isinstance(content, str):
                    converted[-1]["content"] = f"{prev}\n{content}"
                else:
                    if isinstance(prev, str):
                        prev = [{"type": "text", "text": prev}]
                    if isinstance(content, str):
                        content = [{"type": "text", "text": content}]
                    converted[-1]["content"] = prev + content
            else:
                converted.append({"role": role, "content": content})

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if role == "system":
                text = self._content_to_text(content)
                if text:
                    system_parts.append(text)
            elif role == "tool":
                append("user", [{
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": self._content_to_text(content),
                }])
            elif role == "assistant" and msg.get("tool_calls"):
                blocks = []
                text = self._content_to_text(content)
                if text:
                    blocks.append({"type": "text", "text": text})
                for tc in msg["tool_calls"]:
                    if isinstance(tc, dict):
                        fn = tc.get("function", {})
                        name = fn.get("name")
                        arguments = fn.get("arguments", "{}")
                    else:
                        name = tc.function.name
                        arguments = tc.function.arguments
                    input_data = json.loads(arguments) if isinstance(arguments, str) else arguments
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id") if isinstance(tc, dict) else tc.id,
                        "name": name,
                        "input": input_data or {},
                    })
                append("assistant", blocks)
            else:
                append("user" if role == "user" else "assistant", self._content_to_text(content))

        return "\n\n".join(system_parts) or None, converted

    def _anthropic_tools(self, tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        converted = []
        for tool in tools or []:
            if tool.get("type") == "function" and "function" in tool:
                fn = tool["function"]
                converted.append({
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
                })
            else:
                converted.append(tool)
        return converted

    def _anthropic_tool_choice(self, tool_choice: Any) -> Optional[Dict[str, Any]]:
        if tool_choice is None:
            return None
        if tool_choice == "auto":
            return {"type": "auto"}
        if tool_choice == "none":
            return {"type": "none"}
        if tool_choice in ("any", "required"):
            return {"type": "any"}
        if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            return {"type": "tool", "name": tool_choice["function"]["name"]}
        return None

    def _anthropic_params(self, messages, tools, merged_params, stream=False):
        system, converted = self._anthropic_messages(messages)
        params = {
            "model": self.config.model_id,
            "messages": converted,
            "max_tokens": merged_params.pop("max_tokens", 4096),
        }
        if system:
            params["system"] = system
        if "temperature" in merged_params:
            params["temperature"] = merged_params.pop("temperature")
        converted_tools = self._anthropic_tools(tools)
        if converted_tools:
            params["tools"] = converted_tools
            tool_choice = self._anthropic_tool_choice(merged_params.pop("tool_choice", None))
            if tool_choice:
                params["tool_choice"] = tool_choice
        params.update(merged_params)
        if stream:
            params["stream"] = True
        return params

    def _map_stop_reason(self, stop_reason: Any) -> Optional[str]:
        return {
            "end_turn": "stop",
            "stop_sequence": "stop",
            "max_tokens": "length",
            "tool_use": "tool_calls",
            "refusal": "content_filter",
        }.get(stop_reason, stop_reason)

    def _generate_anthropic(self, messages, tools, merged_params):
        response = self.client.messages.create(
            **self._anthropic_params(messages, tools, merged_params)
        )
        text_parts = []
        tool_calls = []
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(block.text)
            elif block_type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input) if not isinstance(block.input, str) else block.input,
                    },
                })
        return {
            "content": "".join(text_parts) or None,
            "role": "assistant",
            "tool_calls": tool_calls or None,
            "finish_reason": self._map_stop_reason(response.stop_reason),
            "usage": self._anthropic_usage(response.usage),
            "model": response.model,
        }

    def _responses_params(self, messages, tools, merged_params, stream=False):
        params = {"model": self.config.model_id, "input": messages}
        if stream:
            params["stream"] = True
        if "max_tokens" in merged_params:
            params["max_output_tokens"] = merged_params.pop("max_tokens")
        if "response_format" in merged_params:
            response_format = merged_params.pop("response_format")
            if response_format.get("type") == "json_schema":
                response_format = {"type": "json_schema", **response_format["json_schema"]}
            params["text"] = {"format": response_format}
        if "reasoning_effort" in merged_params:
            params["reasoning"] = {"effort": merged_params.pop("reasoning_effort")}
        if tools:
            params["tools"] = [
                {"type": "function", **tool["function"]}
                if tool.get("type") == "function" and "function" in tool else tool
                for tool in tools
            ]
            if "tool_choice" in merged_params:
                params["tool_choice"] = merged_params.pop("tool_choice")
        params.update(merged_params)
        return params

    def _generate_responses(self, messages, tools, merged_params):
        response = self.client.responses.create(
            **self._responses_params(messages, tools, merged_params)
        )
        tool_calls = []
        for item in response.output:
            if getattr(item, "type", None) == "function_call":
                tool_calls.append({
                    "id": item.call_id,
                    "type": "function",
                    "function": {"name": item.name, "arguments": item.arguments},
                })
        return {
            "content": response.output_text or None,
            "role": "assistant",
            "tool_calls": tool_calls or None,
            "finish_reason": response.status,
            "usage": self._usage_to_dict(response.usage),
            "model": response.model,
            "response_id": response.id,
        }
    
    def _generate_impl(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        merged_params = self._merge_params(**kwargs)
        if self._is_anthropic():
            return self._generate_anthropic(messages, tools, merged_params)
        if self.config.api_type == LLMApiType.RESPONSES:
            return self._generate_responses(messages, tools, merged_params)
        
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
            if isinstance(tc, dict):
                result.append({
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                })
                continue
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
        if self._is_anthropic():
            raise ValueError("Embedding is not supported by the Anthropic API")
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
        if self._is_anthropic():
            raise ValueError("Perplexity is not supported by the Anthropic API")
        if self.config.api_type == LLMApiType.RESPONSES:
            raise ValueError("Perplexity is not supported by the Responses API")
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
        if self._is_anthropic():
            self.last_stream_usage = None
            prompt_tokens = 0
            stream = self.client.messages.create(
                **self._anthropic_params(messages, tools, merged_params, stream=True)
            )
            for event in stream:
                event_type = getattr(event, "type", None)
                if event_type == "message_start":
                    usage = getattr(event.message, "usage", None)
                    if usage:
                        input_tokens = getattr(usage, "input_tokens", 0) or 0
                        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
                        cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
                        prompt_tokens = input_tokens + cache_read + cache_creation
                elif event_type == "content_block_delta":
                    delta = event.delta
                    if getattr(delta, "type", None) == "text_delta":
                        yield delta.text
                elif event_type == "message_delta":
                    usage = getattr(event, "usage", None)
                    output_tokens = getattr(usage, "output_tokens", 0) or 0 if usage else 0
                    self.last_stream_usage = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": output_tokens,
                        "total_tokens": prompt_tokens + output_tokens,
                    }
            return

        if self.config.api_type == LLMApiType.RESPONSES:
            self.last_stream_usage = None
            stream = self.client.responses.create(
                **self._responses_params(messages, tools, merged_params, stream=True)
            )
            for event in stream:
                if event.type == "response.output_text.delta":
                    yield event.delta
                elif event.type == "response.completed":
                    self.last_stream_usage = self._usage_to_dict(event.response.usage)
            return
        
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
