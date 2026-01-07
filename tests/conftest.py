import pytest
import configparser
import os
from llmpivot.config import PivotConfig


def load_test_config():
    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(__file__), 'local_config.ini')
    
    if os.path.exists(config_path):
        config.read(config_path, encoding='utf-8')
    
    return config


@pytest.fixture
def test_config():
    return load_test_config()


@pytest.fixture
def online_config(test_config):
    if test_config.has_section('online'):
        return PivotConfig(
            model_type="online",
            model_id=test_config.get('online', 'model_id'),
            api_key=test_config.get('online', 'api_key'),
            base_url=test_config.get('online', 'base_url'),
            enable_log=test_config.getboolean('online', 'enable_log', fallback=False)
        )
    else:
        return PivotConfig(
            model_type="online",
            model_id="gpt-4",
            api_key="test-api-key",
            base_url="https://api.openai.com/v1",
            enable_log=False
        )


@pytest.fixture
def local_config(test_config):
    if test_config.has_section('local'):
        return PivotConfig(
            model_type="local",
            model_id=test_config.get('local', 'model_id'),
            base_url=test_config.get('local', 'base_url'),
            enable_log=test_config.getboolean('local', 'enable_log', fallback=False)
        )
    else:
        return PivotConfig(
            model_type="local",
            model_id="qwen2.5:7b",
            base_url="http://localhost:11434",
            enable_log=False
        )


@pytest.fixture
def sample_messages():
    return [
        {"role": "user", "content": "Hello, how are you?"}
    ]


@pytest.fixture
def sample_tools():
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
