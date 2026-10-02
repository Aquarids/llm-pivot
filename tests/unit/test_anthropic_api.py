from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from llmpivot.config import PivotConfig
from llmpivot.llm.api_llm import ApiLLM


@pytest.fixture
def anthropic_llm():
    config = PivotConfig(
        model_type="online",
        api_type="anthropic",
        model_id="glm-5.3-flash",
        api_key="test-key",
        base_url="http://localhost:18400/v1",
        enable_log=False,
    )
    with patch("llmpivot.llm.api_llm.Anthropic") as client_class:
        client = client_class.return_value
        yield ApiLLM(config, Mock()), client


def test_chat_completions_uses_openai_client():
    config = PivotConfig(
        model_type="online",
        model_id="gpt-test",
        api_key="test-key",
        enable_log=False,
    )
    with patch("llmpivot.llm.api_llm.OpenAI") as openai_class, \
            patch("llmpivot.llm.api_llm.Anthropic") as anthropic_class:
        ApiLLM(config, Mock())
        openai_class.assert_called_once()
        anthropic_class.assert_not_called()


def test_client_strips_v1_suffix_and_uses_bearer_auth():
    config = PivotConfig(
        model_type="online",
        api_type="anthropic",
        model_id="glm-5.3-flash",
        api_key="test-key",
        base_url="http://localhost:18400/v1",
        enable_log=False,
    )
    with patch("llmpivot.llm.api_llm.Anthropic") as client_class:
        ApiLLM(config, Mock())
        kwargs = client_class.call_args.kwargs
        assert kwargs["auth_token"] == "test-key"
        assert kwargs["base_url"] == "http://localhost:18400"


def test_generate_maps_system_and_text_blocks(anthropic_llm):
    llm, client = anthropic_llm
    client.messages.create.return_value = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="pondering"),
            SimpleNamespace(type="text", text="OK"),
        ],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=60, output_tokens=37, cache_read_input_tokens=256),
        model="glm-5.3-flash",
    )

    result = llm._generate_impl([
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "Reply with exactly: OK"},
    ], max_tokens=100, temperature=0.5)

    params = client.messages.create.call_args.kwargs
    assert params["model"] == "glm-5.3-flash"
    assert params["system"] == "Be terse."
    assert params["messages"] == [{"role": "user", "content": "Reply with exactly: OK"}]
    assert params["max_tokens"] == 100
    assert params["temperature"] == 0.5
    assert "stream" not in params
    assert result["content"] == "OK"
    assert result["tool_calls"] is None
    assert result["finish_reason"] == "stop"
    assert result["usage"] == {
        "prompt_tokens": 316,
        "completion_tokens": 37,
        "total_tokens": 353,
    }
    assert result["model"] == "glm-5.3-flash"


def test_generate_applies_default_max_tokens_without_temperature(anthropic_llm):
    llm, client = anthropic_llm
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="hi")],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        model="glm-5.3-flash",
    )

    llm._generate_impl([{"role": "user", "content": "hi"}])

    params = client.messages.create.call_args.kwargs
    assert params["max_tokens"] == 4096
    assert "temperature" not in params


def test_generate_converts_tools_and_tool_use_blocks(anthropic_llm):
    llm, client = anthropic_llm
    client.messages.create.return_value = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="Checking."),
            SimpleNamespace(type="tool_use", id="toolu_1", name="get_weather", input={"city": "Beijing"}),
        ],
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        model="glm-5.3-flash",
    )
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }]

    result = llm._generate_impl([{"role": "user", "content": "Weather in Beijing?"}], tools)

    params = client.messages.create.call_args.kwargs
    assert params["tools"] == [{
        "name": "get_weather",
        "description": "Get the current weather",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
    }]
    assert result["tool_calls"] == [{
        "id": "toolu_1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'},
    }]
    assert result["finish_reason"] == "tool_calls"


def test_call_function_normalizes_anthropic_tool_call(anthropic_llm):
    llm, client = anthropic_llm
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", id="toolu_1", name="get_weather", input={"city": "Shanghai"})],
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        model="glm-5.3-flash",
    )

    result = llm._call_function_impl(
        [{"role": "user", "content": "Weather?"}],
        [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
        tool_choice="auto",
    )

    assert client.messages.create.call_args.kwargs["tool_choice"] == {"type": "auto"}
    assert result == [{
        "id": "toolu_1",
        "name": "get_weather",
        "arguments": '{"city": "Shanghai"}',
    }]


def test_tool_result_and_assistant_tool_calls_are_converted(anthropic_llm):
    llm, client = anthropic_llm
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="Sunny")],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=10, output_tokens=2),
        model="glm-5.3-flash",
    )
    messages = [
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "toolu_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'},
        }]},
        {"role": "tool", "tool_call_id": "toolu_1", "content": "Sunny, 25C"},
    ]

    llm._generate_impl(messages)

    params = client.messages.create.call_args.kwargs
    assert params["messages"][1] == {
        "role": "assistant",
        "content": [{
            "type": "tool_use",
            "id": "toolu_1",
            "name": "get_weather",
            "input": {"city": "Beijing"},
        }],
    }
    assert params["messages"][2] == {
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": "toolu_1",
            "content": "Sunny, 25C",
        }],
    }


def test_stream_generate_yields_text_deltas_and_usage(anthropic_llm):
    llm, client = anthropic_llm
    client.messages.create.return_value = iter([
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(usage=SimpleNamespace(input_tokens=60, cache_read_input_tokens=256)),
        ),
        SimpleNamespace(type="content_block_start", index=0),
        SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="thinking_delta", thinking="hmm")),
        SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="text_delta", text="OK")),
        SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="end_turn"),
            usage=SimpleNamespace(output_tokens=37),
        ),
        SimpleNamespace(type="message_stop"),
    ])

    chunks = list(llm._generate_stream_impl([{"role": "user", "content": "Reply OK"}], max_tokens=100))

    assert chunks == ["OK"]
    assert client.messages.create.call_args.kwargs["stream"] is True
    assert llm.last_stream_usage == {
        "prompt_tokens": 316,
        "completion_tokens": 37,
        "total_tokens": 353,
    }


def test_dialogue_uses_anthropic_path(anthropic_llm):
    llm, client = anthropic_llm
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="2+2=4")],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=5, output_tokens=3),
        model="glm-5.3-flash",
    )

    assert llm._dialogue_impl([{"role": "user", "content": "What is 2+2?"}]) == "2+2=4"


def test_perplexity_is_rejected_for_anthropic(anthropic_llm):
    llm, _ = anthropic_llm
    with pytest.raises(ValueError, match="not supported"):
        llm._perplexity_impl("hello")


def test_embedding_is_rejected_for_anthropic(anthropic_llm):
    llm, _ = anthropic_llm
    with pytest.raises(ValueError, match="not supported"):
        llm._embedding_impl("hello")
