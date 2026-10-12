import json
from unittest.mock import patch

import httpx
import pytest
from openai import OpenAI

from llmpivot import LLMPivot, PivotConfig, SIWCAuth, SIWCConfig
from llmpivot.siwc.auth import SIWCError
from llmpivot.siwc.cli import main


def completed(output=None, **kwargs):
    return {"type": "response.completed", "sequence_number": 3, "response": {
        "id": "resp_test", "object": "response", "created_at": 1, "status": "completed",
        "model": "visible-model", "output": output or [],
        "usage": {"input_tokens": 7, "output_tokens": 2, "total_tokens": 9}, **kwargs}}


def delta(text="SIWC_OK"):
    return {"type": "response.output_text.delta", "sequence_number": 1, "item_id": "msg_test",
            "output_index": 0, "content_index": 0, "delta": text, "logprobs": []}


@pytest.fixture
def backend():
    events = [delta(), completed()]
    requests = []
    status = [200]
    def handler(request):
        requests.append(request)
        if status[0] != 200:
            return httpx.Response(status[0], json={"error": {"code": "test_denied", "message": "secret-token-sentinel"}})
        text = "".join("event: " + e["type"] + "\ndata: " + json.dumps(e) + "\n\n" for e in events)
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, text=text)
    def factory(**kwargs):
        assert kwargs["max_retries"] == 0
        return OpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    with patch("llmpivot.llm.siwc_llm.OpenAI", side_effect=factory), \
         patch("llmpivot.llm.siwc_llm.get_access_token", return_value="fake-access") as token:
        with LLMPivot(PivotConfig(api_type="siwc", model_id="visible-model", enable_log=False)) as pivot:
            yield pivot, requests, events, status, token


def test_public_generate_no_key_and_normalized_result(backend):
    pivot, requests, _, _, _ = backend
    result = pivot.generate([{"role": "system", "content": "Instruction"}, {"role": "user", "content": "Hello"}])
    body = json.loads(requests[0].content)
    assert str(requests[0].url) == "https://api.openai.com/v1/responses"
    assert body["store"] is False and body["stream"] is True
    assert body["input"][0]["role"] == "developer"
    assert result["content"] == "SIWC_OK"
    assert result["finish_reason"] == "completed"
    assert result["response_id"] == "resp_test"
    assert result["usage"] == {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}


@pytest.mark.parametrize("stream", [False, True])
def test_dialogue_interfaces(backend, stream):
    pivot, *_ = backend
    assert pivot.dialogue([{"role": "user", "content": "Hello"}], stream=stream) == "SIWC_OK"
    result = pivot.dialogue_with_usage([{"role": "user", "content": "Hello"}], stream=stream)
    assert result["content"] == "SIWC_OK" and result["usage"]["total_tokens"] == 9


def test_stream_and_fresh_token_each_request(backend):
    pivot, requests, _, _, token = backend
    token.side_effect = ["fake-one", "fake-two"]
    assert list(pivot.stream_generate([{"role": "user", "content": "Hello"}])) == ["SIWC_OK"]
    pivot.generate([{"role": "user", "content": "Again"}])
    assert [r.headers["Authorization"] for r in requests] == ["Bearer fake-one", "Bearer fake-two"]
    assert pivot.llm.last_stream_usage["total_tokens"] == 9


@pytest.mark.parametrize("parameter,value", [("temperature", 0.1), ("max_tokens", 20),
    ("max_output_tokens", 20), ("previous_response_id", "resp_old"), ("store", True),
    ("stream", False), ("metadata", {}), ("input", [])])
def test_invalid_parameters_rejected_before_token(backend, parameter, value):
    pivot, requests, _, _, token = backend
    with pytest.raises(ValueError):
        pivot.generate([{"role": "user", "content": "Hello"}], **{parameter: value})
    assert not requests
    token.assert_not_called()


@pytest.mark.parametrize("terminal", [None, "response.failed", "response.incomplete", "error"])
def test_stream_failure_does_not_replay_or_return_partial(backend, terminal):
    pivot, requests, events, _, _ = backend
    events[:] = [delta("partial")]
    if terminal:
        event = {"type": terminal, "sequence_number": 2}
        if terminal != "error":
            event["response"] = {**completed()["response"], "status": terminal.split(".")[1]}
        events.append(event)
    with pytest.raises(SIWCError):
        pivot.dialogue([{"role": "user", "content": "Hello"}], stream=True)
    assert len(requests) == 1


def test_live_stream_surfaces_failure_after_delta(backend):
    pivot, _, events, _, _ = backend
    events[:] = [delta("partial")]
    stream = pivot.stream_generate([{"role": "user", "content": "Hello"}])
    assert next(stream) == "partial"
    with pytest.raises(SIWCError, match="without response.completed"):
        next(stream)


def test_http_errors_are_sanitized_and_not_retried(backend):
    pivot, requests, _, status, _ = backend
    status[0] = 403
    with pytest.raises(SIWCError) as error:
        pivot.generate([{"role": "user", "content": "Hello"}])
    assert "403" in str(error.value) and "test_denied" in str(error.value)
    assert "secret-token-sentinel" not in str(error.value)
    assert len(requests) == 1


def test_function_tools_and_history_are_normalized(backend):
    pivot, requests, events, _, _ = backend
    call = {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "lookup",
            "arguments": '{"q":"test"}', "namespace": "llm_pivot", "status": "completed"}
    events[:] = [{"type": "response.output_item.done", "sequence_number": 1, "output_index": 0, "item": call}, completed()]
    messages = [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "old_call", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "old_call", "content": "result"}]
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}}}]
    result = pivot.generate(messages, tools=tools)
    body = json.loads(requests[0].content)
    assert body["tools"][0]["type"] == "namespace"
    assert body["tools"][0]["tools"][0]["name"] == "lookup"
    assert body["input"][0]["type"] == "function_call"
    assert body["input"][1]["type"] == "function_call_output"
    assert result["tool_calls"][0]["namespace"] == "llm_pivot"
    assert pivot.call_function([{"role": "user", "content": "Lookup"}], tools)[0]["name"] == "lookup"


def test_structured_output_and_reasoning_mapping(backend):
    pivot, requests, *_ = backend
    pivot.generate([{"role": "user", "content": "Hello"}], response_format={"type": "json_object"}, reasoning_effort="low")
    body = json.loads(requests[0].content)
    assert body["text"]["format"] == {"type": "json_object"}
    assert body["reasoning"] == {"effort": "low"}


def test_reject_alternative_credentials_and_endpoints():
    for kwargs in ({"api_key": "user-supplied"}, {"base_url": "https://unrelated.example/v1"}):
        with pytest.raises(ValueError):
            LLMPivot(PivotConfig(api_type="siwc", enable_log=False, **kwargs))


def test_unsupported_operations(backend):
    pivot, requests, *_ = backend
    for operation in (pivot.embedding, pivot.perplexity):
        with pytest.raises(ValueError, match="not supported"):
            operation("text")
    with pytest.raises(ValueError, match="tool type"):
        pivot.generate([], tools=[{"type": "image_generation"}])
    assert not requests


def test_cli_status_does_not_print_credentials(tmp_path, capsys):
    from llmpivot.siwc.auth import CredentialStore
    store = CredentialStore(tmp_path)
    with store.locked():
        store.write({"access_token": "secret-token-sentinel", "refresh_token": "secret-refresh-sentinel", "subject": "private-subject", "scopes": []})
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"api_type": "siwc", "siwc": {"credentials_dir": str(tmp_path)}}))
    assert main(["--config", str(config_path), "status"]) == 0
    output = capsys.readouterr().out
    assert "Signed in: True" in output and "plan permission: False" in output
    assert "sentinel" not in output and "private-subject" not in output


def test_config_file_and_separate_authorization_api(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"api_type": "siwc", "model_id": "visible-model", "siwc": {
        "app_name": "My Experiments", "credentials_dir": str(tmp_path / "credentials"),
        "login_timeout": 42, "refresh_margin": 30, "max_concurrency": 3}}))
    config = PivotConfig.from_file(path)
    auth = SIWCAuth(config.siwc)
    with patch("llmpivot.siwc.auth.login") as login:
        auth.login(consent=True)
        login.assert_called_once_with(auth.store, config.siwc, consent=True)
    assert auth.config.app_name == "My Experiments"
    assert auth.config.login_timeout == 42
    assert auth.config.max_concurrency == 3


def test_distinct_app_names_have_distinct_default_stores(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("LLM_PIVOT_SIWC_DIR", raising=False)
    first = SIWCAuth(SIWCConfig(app_name="App One")).store.directory
    second = SIWCAuth(SIWCConfig(app_name="App Two")).store.directory
    assert first != second
    assert first == SIWCAuth(SIWCConfig(app_name="App One")).store.directory


def test_async_concurrency_and_configured_limit():
    import asyncio
    import threading
    import time
    lock = threading.Lock()
    barrier = threading.Barrier(2, timeout=5)
    active = 0
    peak = 0
    def request(messages, tools, kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            barrier.wait()
            time.sleep(0.03)
            yield messages[0]["content"]
            return {"content": messages[0]["content"], "usage": None}
        finally:
            with lock:
                active -= 1
    async def run(pivot):
        return await asyncio.gather(*[pivot.agenerate([{"role": "user", "content": str(i)}]) for i in range(6)])
    with LLMPivot(PivotConfig(api_type="siwc", enable_log=False, siwc=SIWCConfig(max_concurrency=2))) as pivot:
        with patch.object(pivot.llm, "_run_request", side_effect=request):
            results = asyncio.run(run(pivot))
    assert [r["content"] for r in results] == [str(i) for i in range(6)]
    assert peak == 2 and active == 0
