import os
import configparser
from llmpivot import LLMPivot, PivotConfig
import pytest


@pytest.fixture(scope="module")
def test_config():
    config = configparser.ConfigParser()
    project_root = os.path.dirname(os.path.dirname(__file__))
    
    local_config_path = os.path.join(project_root, 'local_config.ini')
    default_config_path = os.path.join(project_root, 'default_config.ini')
    
    if os.path.exists(local_config_path):
        config.read(local_config_path, encoding='utf-8')
    else:
        config.read(default_config_path, encoding='utf-8')
    
    return config


@pytest.fixture(scope="module")
def api_config(test_config):
    if not test_config.has_section('online'):
        pytest.skip("No online configuration found")
    
    return PivotConfig(
        model_type="online",
        model_id=test_config.get('online', 'model_id'),
        api_key=test_config.get('online', 'api_key'),
        base_url=test_config.get('online', 'base_url'),
        enable_log=test_config.getboolean('online', 'enable_log', fallback=False)
    )


@pytest.fixture
def weather_tool():
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city name"
                        }
                    },
                    "required": ["city"]
                }
            }
        }
    ]


class TestAPILLMIntegration:
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_generate(self, api_config):
        with LLMPivot(api_config) as llm:
            messages = [
                {"role": "user", "content": "Hello, please introduce yourself briefly."}
            ]
            
            response = llm.generate(messages)
            print(response)
            
            assert response is not None
            assert 'content' in response or 'tool_calls' in response
            if 'content' in response:
                assert isinstance(response['content'], str)
                assert len(response['content']) > 0
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_dialogue(self, api_config):
        with LLMPivot(api_config) as llm:
            messages = [
                {"role": "user", "content": "What is 2+2?"}
            ]
            
            response = llm.dialogue(messages)
            print(response)
            
            assert response is not None
            assert isinstance(response, str)
            assert len(response) > 0
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_with_tools(self, api_config, weather_tool):
        with LLMPivot(api_config) as llm:
            messages = [
                {"role": "user", "content": "What's the weather like in Beijing?"}
            ]
            
            response = llm.generate(messages, tools=weather_tool)
            
            assert response is not None
            assert 'content' in response or 'tool_calls' in response
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_call_function(self, api_config, weather_tool):
        with LLMPivot(api_config) as llm:
            messages = [
                {"role": "user", "content": "Check the weather in Shanghai"}
            ]
            
            tool_calls = llm.call_function(messages, weather_tool)
            print(tool_calls)
            
            assert tool_calls is not None
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_stream_generate(self, api_config):
        with LLMPivot(api_config) as llm:
            messages = [
                {"role": "user", "content": "Count from 1 to 3"}
            ]
            
            chunks = []
            for chunk in llm.stream_generate(messages):
                assert isinstance(chunk, str)
                chunks.append(chunk)
            
            full_response = ''.join(chunks)
            print(full_response)
            assert len(full_response) > 0
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_multi_turn_conversation(self, api_config):
        with LLMPivot(api_config) as llm:
            messages = [
                {"role": "user", "content": "My name is Alice."},
            ]
            
            response1 = llm.dialogue(messages)
            print(response1)
            assert response1 is not None
            
            messages.append({"role": "assistant", "content": response1})
            messages.append({"role": "user", "content": "What's my name?"})
            
            response2 = llm.dialogue(messages)
            print(response2)
            assert response2 is not None
            assert "alice" in response2.lower()
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_context_manager(self, api_config):
        with LLMPivot(api_config) as llm:
            assert llm is not None
            messages = [{"role": "user", "content": "Hi"}]
            response = llm.dialogue(messages)
            assert response is not None
