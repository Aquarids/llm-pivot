# tests/base_test.py
import sys
import torch
import os
import configparser

from src.llmpivot.helper import Logger
from src.llmpivot import LLMPivot, PivotConfig

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

project_root = os.path.dirname(__file__)
logger = Logger(__name__, 'dev')

def read_config(local_config_path, default_config_path):
    config = configparser.ConfigParser()

    if os.path.exists(local_config_path):
        config.read(local_config_path, encoding='utf-8')
        print(f'Loaded configuration from {local_config_path}')
    else:
        config.read(default_config_path, encoding='utf-8')
        print(f'Loaded configuration from {default_config_path}')

    return config

test_config = read_config(
    os.path.join(project_root, 'tests', 'local_config.ini'),
    os.path.join(project_root, 'tests', 'default_config.ini'),
)

weather_tool = [
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

api_config = PivotConfig(
    model_type="online",
    model_id=test_config.get('online', 'model_id'),
    api_key=test_config.get('online', 'api_key'),
    base_url=test_config.get('online', 'base_url'),
    enable_log=test_config.getboolean('online', 'enable_log', fallback=False)
)

if test_config.has_section('local'):
    local_config = PivotConfig(
        model_type="local",
        model_id=test_config.get('local', 'model_id'),
        base_url=test_config.get('local', 'base_url'),
        enable_log=test_config.getboolean('local', 'enable_log', fallback=False)
    )

def test_api_llm_generate():
    print("\n=== Testing API LLM Generate ===")
    llm = LLMPivot(api_config)
    
    messages = [
        {"role": "user", "content": "Hello, please introduce yourself briefly."}
    ]
    
    response = llm.generate(messages)
    print(f"Response: {response}")
    assert response is not None
    assert 'content' in response or 'tool_calls' in response
    print("✓ API LLM Generate test passed")


def test_api_llm_dialogue():
    print("\n=== Testing API LLM Dialogue ===")
    llm = LLMPivot(api_config)
    
    messages = [
        {"role": "user", "content": "What is 2+2?"}
    ]
    
    response = llm.dialogue(messages)
    print(f"Dialogue response: {response}")
    assert response is not None
    assert isinstance(response, str)
    print("✓ API LLM Dialogue test passed")


def test_api_llm_with_tools():
    print("\n=== Testing API LLM with Tools ===")
    llm = LLMPivot(api_config)
    
    messages = [
        {"role": "user", "content": "What's the weather like in Beijing?"}
    ]
    
    response = llm.generate(messages, tools=weather_tool)
    print(f"Response with tools: {response}")
    assert response is not None
    print("✓ API LLM with Tools test passed")


def test_api_llm_stream():
    print("\n=== Testing API LLM Stream Generate ===")
    llm = LLMPivot(api_config)
    
    messages = [
        {"role": "user", "content": "Count from 1 to 5"}
    ]
    
    print("Stream output: ", end='')
    full_response = ""
    for chunk in llm.stream_generate(messages):
        print(chunk, end='', flush=True)
        full_response += chunk
    print()
    
    assert len(full_response) > 0
    print("✓ API LLM Stream Generate test passed")


def test_local_llm_generate():
    print("\n=== Testing Local LLM Generate ===")
    llm = LLMPivot(local_config)
    
    messages = [
        {"role": "user", "content": "Hello, please introduce yourself briefly."}
    ]
    
    response = llm.generate(messages)
    print(f"Response: {response}")
    assert response is not None
    assert 'content' in response or 'tool_calls' in response
    print("✓ Local LLM Generate test passed")


def test_local_llm_dialogue():
    print("\n=== Testing Local LLM Dialogue ===")
    llm = LLMPivot(local_config)
    
    messages = [
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    response = llm.dialogue(messages)
    print(f"Dialogue response: {response}")
    assert response is not None
    assert isinstance(response, str)
    print("✓ Local LLM Dialogue test passed")


def test_context_manager():
    print("\n=== Testing Context Manager ===")
    
    with LLMPivot(api_config) as llm:
        messages = [{"role": "user", "content": "Hi"}]
        response = llm.dialogue(messages)
        print(f"Response in context: {response}")
        assert response is not None
    
    print("✓ Context Manager test passed")


if __name__ == "__main__":
    try:
        test_api_llm_generate()
        test_api_llm_dialogue()
        test_api_llm_with_tools()
        test_api_llm_stream()
        
        if test_config.has_section('local'):
            test_local_llm_generate()
            test_local_llm_dialogue()
        
        test_context_manager()
        
        print("\n" + "="*50)
        print("✓ All tests passed!")
        print("="*50)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
