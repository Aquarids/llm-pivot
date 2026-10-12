from contextlib import closing
from threading import BoundedSemaphore

from openai import OpenAI, APIStatusError, APIConnectionError

from .api_llm import ApiLLM
from .base_llm import BaseLLM
from ..siwc.auth import CredentialStore, RESOURCE, SIWCError, get_access_token, safe_code

UNSUPPORTED = {
    "background", "conversation", "max_tokens", "max_output_tokens", "max_tool_calls",
    "metadata", "moderation", "multi_agent", "prompt", "prompt_cache_retention",
    "safety_identifier", "temperature", "top_logprobs", "top_p", "truncation", "user",
    "previous_response_id", "stream_options", "logprobs", "stop", "input", "model",
}


class SIWCLLM(ApiLLM):
    def __init__(self, config, logger):
        BaseLLM.__init__(self, config, logger)
        if config.api_key is not None:
            raise ValueError("SIWC manages OAuth credentials; omit api_key.")
        if config.base_url and config.base_url.rstrip("/") != RESOURCE:
            raise ValueError("SIWC uses https://api.openai.com/v1; omit base_url.")
        self.store = CredentialStore(config.siwc.credentials_dir, app_name=config.siwc.app_name)
        self._slots = BoundedSemaphore(config.siwc.max_concurrency)
        self.last_stream_usage = None

    def _retry_wrapper(self, func, *args, **kwargs):

        return func(*args, **kwargs)

    def dialogue(self, messages, stream=False, **kwargs):
        return self._dialogue_impl(messages, **kwargs)

    def dialogue_with_usage(self, messages, stream=False, **kwargs):
        result = self._generate_impl(messages, None, **kwargs)
        return {"content": result["content"] or "", "usage": result["usage"]}

    def _input(self, messages):
        converted = []
        for message in messages:
            item = dict(message)
            if item.get("role") == "system":
                item["role"] = "developer"
            if item.get("role") == "tool":
                converted.append({"type": "function_call_output", "call_id": item["tool_call_id"],
                                  "output": item.get("content") or ""})
            elif item.get("role") == "assistant" and item.get("tool_calls"):
                if item.get("content"):
                    converted.append({"role": "assistant", "content": item["content"]})
                for call in item["tool_calls"]:
                    fn = call["function"]
                    converted.append({"type": "function_call", "call_id": call["id"],
                                      "name": fn["name"], "arguments": fn["arguments"],
                                      "namespace": call.get("namespace", "llm_pivot")})
            else:
                converted.append(item)
        return converted

    def _params(self, messages, tools, kwargs):
        merged = self._merge_params(**kwargs)
        unsupported = UNSUPPORTED.intersection(merged)
        if unsupported:
            raise ValueError("SIWC does not support parameters: " + ", ".join(sorted(unsupported)))
        if merged.pop("store", False) is not False or merged.pop("stream", True) is not True:
            raise ValueError("SIWC requires store=False and stream=True on HTTP requests.")
        if merged.get("tool_choice", "auto") not in ("auto", "required", "none"):
            raise ValueError("SIWC tool_choice currently supports auto, required or none.")
        params = self._responses_params(self._input(messages), tools, merged, stream=True)
        params["store"] = False

        direct, grouped = [], []
        for tool in params.get("tools", []):
            kind = tool.get("type")
            if kind == "function":
                direct.append(tool)
            elif kind == "namespace":
                if any(t.get("type") != "function" for t in tool.get("tools", [])):
                    raise ValueError("SIWC namespaces currently support function tools only.")
                grouped.append(tool)
            elif kind in {"web_search", "web_search_preview"}:
                grouped.append(tool)
            else:
                raise ValueError("Unsupported SIWC tool type: " + str(kind))
        if direct:
            grouped.append({"type": "namespace", "name": "llm_pivot",
                            "description": "Research application's local tools", "tools": direct})
        if grouped:
            params["tools"] = grouped
        return params

    def _run(self, messages, tools, kwargs):
        with self._slots:
            return (yield from self._run_request(messages, tools, kwargs))

    def _run_request(self, messages, tools, kwargs):
        params = self._params(messages, tools, kwargs)
        token = get_access_token(self.store, refresh_margin=self.config.siwc.refresh_margin)
        self.last_stream_usage = None
        completed, chunks, calls = None, [], {}
        try:

            with OpenAI(api_key=token, base_url=RESOURCE, timeout=self.config.timeout, max_retries=0) as client:
                with client.responses.create(**params) as stream:
                    for event in stream:
                        if event.type == "response.output_text.delta":
                            chunks.append(event.delta)
                            yield event.delta
                        elif event.type == "response.output_item.done":
                            item = event.item
                            if item.type == "function_call":
                                calls[item.call_id] = item
                        elif event.type == "response.completed":
                            completed = event.response
                        elif event.type in {"response.failed", "response.incomplete", "error"}:
                            error = getattr(getattr(event, "response", None), "error", None)
                            code = getattr(error, "code", None) or getattr(event, "code", None)
                            raise SIWCError(f"{event.type}: code={safe_code(code)}; usage: https://chatgpt.com/settings/usage")
            if completed is None or completed.status != "completed":
                raise SIWCError("Stream ended without response.completed; result is incomplete.")
            for item in completed.output:
                if item.type == "function_call":
                    calls[item.call_id] = item
            content = completed.output_text or "".join(chunks)
            normalized = []
            for call in calls.values():
                tool = {"id": call.call_id, "type": "function",
                        "function": {"name": call.name, "arguments": call.arguments}}
                if getattr(call, "namespace", None):
                    tool["namespace"] = call.namespace
                normalized.append(tool)
            usage = self._usage_to_dict(completed.usage) if completed.usage is not None else None
            self.last_stream_usage = usage
            if not chunks and content:
                yield content
            return {"content": content or None, "role": "assistant", "tool_calls": normalized or None,
                    "finish_reason": "completed", "usage": usage, "model": completed.model,
                    "response_id": completed.id}
        except APIStatusError as exc:
            body = exc.body if isinstance(exc.body, dict) else {}
            detail = body.get("error", body)
            code = detail.get("code") if isinstance(detail, dict) else None
            raise SIWCError(f"Inference HTTP {exc.status_code}, code={safe_code(code)}, "
                            f"request_id={safe_code(exc.request_id)}; usage: https://chatgpt.com/settings/usage") from None
        except APIConnectionError:
            raise SIWCError("Inference connection interrupted; completion unverified. No automatic retry.") from None

    def _generate_impl(self, messages, tools=None, **kwargs):
        with closing(self._run(messages, tools, kwargs)) as events:
            while True:
                try:
                    next(events)
                except StopIteration as end:
                    return end.value

    def _generate_stream_impl(self, messages, tools=None, **kwargs):
        yield from self._run(messages, tools, kwargs)

    def embedding(self, text, **kwargs):
        raise ValueError("Embedding is not supported by SIWC.")

    def perplexity(self, text, **kwargs):
        raise ValueError("Perplexity is not supported by SIWC.")

    def _embedding_impl(self, text, **kwargs):
        raise ValueError("Embedding is not supported by SIWC.")

    def _perplexity_impl(self, text, **kwargs):
        raise ValueError("Perplexity is not supported by SIWC.")

    def cleanup(self):

        pass
