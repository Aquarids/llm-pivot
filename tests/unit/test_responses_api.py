from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from llmpivot.config import PivotConfig
from llmpivot.llm.api_llm import ApiLLM


@pytest.fixture
def responses_llm():
    config = PivotConfig(
        model_type="online",
        api_type="responses",
        model_id="gpt-test",
        api_key="test-key",
        enable_log=False,
    )
    with patch("llmpivot.llm.api_llm.OpenAI") as client_class:
        client = client_class.return_value
        yield ApiLLM(config, Mock()), client


def test_chat_completions_remains_the_default():
    assert PivotConfig().api_type == "chat_completions"


def test_generate_maps_responses_request_and_output(responses_llm):
    llm, client = responses_llm
    client.responses.create.return_value = SimpleNamespace(
        id="resp_123",
        output_text="Sunny",
        output=[SimpleNamespace(
            type="function_call",
            call_id="call_123",
            name="get_weather",
            arguments='{"city":"Auckland"}',
        )],
        status="completed",
        usage=SimpleNamespace(input_tokens=10, output_tokens=3, total_tokens=13),
        model="gpt-test",
    )
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {}},
        },
    }]

    result = llm._generate_impl(
        [{"role": "user", "content": "Weather?"}],
        tools,
        max_tokens=50,
        reasoning_effort="low",
    )

    params = client.responses.create.call_args.kwargs
    assert params["input"] == [{"role": "user", "content": "Weather?"}]
    assert "temperature" not in params
    assert params["max_output_tokens"] == 50
    assert params["reasoning"] == {"effort": "low"}
    assert params["tools"][0]["name"] == "get_weather"
    assert "function" not in params["tools"][0]
    assert result["content"] == "Sunny"
    assert result["response_id"] == "resp_123"
    assert result["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 3,
        "total_tokens": 13,
    }


def test_call_function_normalizes_responses_tool_call(responses_llm):
    llm, client = responses_llm
    client.responses.create.return_value = SimpleNamespace(
        id="resp_123",
        output_text="",
        output=[SimpleNamespace(
            type="function_call",
            call_id="call_123",
            name="get_weather",
            arguments='{"city":"Auckland"}',
        )],
        status="completed",
        usage=SimpleNamespace(input_tokens=10, output_tokens=3, total_tokens=13),
        model="gpt-test",
    )

    result = llm._call_function_impl(
        [{"role": "user", "content": "Weather?"}],
        [{"type": "web_search"}],
    )

    assert result == [{
        "id": "call_123",
        "name": "get_weather",
        "arguments": '{"city":"Auckland"}',
    }]


def test_stream_generate_handles_typed_events_and_usage(responses_llm):
    llm, client = responses_llm
    usage = SimpleNamespace(input_tokens=4, output_tokens=2, total_tokens=6)
    client.responses.create.return_value = iter([
        SimpleNamespace(type="response.created"),
        SimpleNamespace(type="response.output_text.delta", delta="Hello"),
        SimpleNamespace(type="response.output_text.delta", delta="!"),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(usage=usage),
        ),
    ])

    assert list(llm._generate_stream_impl([{"role": "user", "content": "Hi"}])) == ["Hello", "!"]
    assert client.responses.create.call_args.kwargs["stream"] is True
    assert llm.last_stream_usage["total_tokens"] == 6


def test_structured_output_is_mapped(responses_llm):
    llm, _ = responses_llm
    params = llm._responses_params([], None, {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "strict": True,
                "schema": {"type": "object"},
            },
        },
    })
    assert params["text"]["format"] == {
        "type": "json_schema",
        "name": "answer",
        "strict": True,
        "schema": {"type": "object"},
    }


def test_perplexity_is_rejected_for_responses(responses_llm):
    llm, _ = responses_llm
    with pytest.raises(ValueError, match="not supported"):
        llm._perplexity_impl("hello")
