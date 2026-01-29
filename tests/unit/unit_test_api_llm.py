import pytest
from unittest.mock import Mock, patch
from llmpivot import LLMPivot
from llmpivot.config import PivotConfig


class TestApiLLM:
    
    def test_init_with_online_model(self, online_config):
        with patch('llmpivot.llmpivot.ApiLLM') as mock_api_llm:

            pivot = LLMPivot(online_config)
            
            print(f"\n[test_init_with_online_model]")
            print(f"Config: {pivot.config}")
            print(f"Model Type: {pivot.config.model_type}")
            print(f"Model ID: {pivot.config.model_id}")
            print(f"Base URL: {pivot.config.base_url}")
            
            assert pivot.config == online_config
            mock_api_llm.assert_called_once()
    
    def test_init_without_api_key_raises_error(self):
        config = PivotConfig(
            model_type="online",
            model_id="gpt-4"
        )
        
        print(f"\n[test_init_without_api_key_raises_error]")
        print(f"Attempting to initialize without API key")
        
        with pytest.raises(ValueError, match="api_key is required"):
            LLMPivot(config)
        
        print(f"ValueError raised as expected")
    
    def test_generate(self, online_config):
        with patch('llmpivot.llmpivot.ApiLLM') as mock_api_llm:
            mock_llm_instance = Mock()
            mock_llm_instance.generate.return_value = {
                "content": "Hello! I'm an AI assistant. How can I help you today?",
                "role": "assistant",
                "usage": {"total_tokens": 20, "prompt_tokens": 5, "completion_tokens": 15}
            }
            mock_api_llm.return_value = mock_llm_instance
            
            pivot = LLMPivot(online_config)
            messages = [{"role": "user", "content": "Hello"}]
            result = pivot.generate(messages)
            
            print(f"\n[test_generate]")
            print(f"Input messages: {messages}")
            print(f"Response: {result}")
            
            mock_llm_instance.generate.assert_called_once_with(messages, None)
    
    def test_generate_with_tools(self, online_config, sample_tools):
        with patch('llmpivot.llmpivot.ApiLLM') as mock_api_llm:
            mock_llm_instance = Mock()
            mock_llm_instance.generate.return_value = {
                "content": None,
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "San Francisco", "unit": "fahrenheit"}'
                        }
                    }
                ]
            }
            mock_api_llm.return_value = mock_llm_instance
            
            pivot = LLMPivot(online_config)
            messages = [{"role": "user", "content": "What's the weather in San Francisco?"}]
            result = pivot.generate(messages, tools=sample_tools)
            
            print(f"\n[test_generate_with_tools]")
            print(f"Input messages: {messages}")
            print(f"Tools provided: {len(sample_tools)} tool(s)")
            print(f"Response: {result}")
            
            mock_llm_instance.generate.assert_called_once_with(messages, sample_tools)
    
    def test_dialogue(self, online_config, sample_messages):
        with patch('llmpivot.llmpivot.ApiLLM') as mock_api_llm:
            mock_llm_instance = Mock()
            mock_llm_instance.dialogue.return_value = "I'm doing great, thank you! I'm here to help you with any questions or tasks you have."
            mock_api_llm.return_value = mock_llm_instance
            
            pivot = LLMPivot(online_config)
            result = pivot.dialogue(sample_messages)
            
            print(f"\n[test_dialogue]")
            print(f"Input messages: {sample_messages}")
            print(f"Response: {result}")
            
            mock_llm_instance.dialogue.assert_called_once_with(sample_messages)
    
    def test_call_function(self, online_config, sample_tools):
        with patch('llmpivot.llmpivot.ApiLLM') as mock_api_llm:
            mock_llm_instance = Mock()
            mock_llm_instance.call_function.return_value = [
                {
                    "id": "call_xyz789",
                    "name": "get_weather",
                    "arguments": '{"city": "New York", "unit": "celsius"}'
                }
            ]
            mock_api_llm.return_value = mock_llm_instance
            
            pivot = LLMPivot(online_config)
            messages = [{"role": "user", "content": "Tell me the weather in New York"}]
            result = pivot.call_function(messages, sample_tools)
            
            print(f"\n[test_call_function]")
            print(f"Input messages: {messages}")
            print(f"Tools: {sample_tools}")
            print(f"Function calls: {result}")
            print(f"Number of function calls: {len(result)}")
            
            mock_llm_instance.call_function.assert_called_once_with(messages, sample_tools)
    
    def test_embedding_single_text(self, online_config):
        with patch('llmpivot.llmpivot.ApiLLM') as mock_api_llm:
            mock_llm_instance = Mock()
            mock_embedding = [0.02, 0.15, -0.08, 0.33, -0.12] + [0.0] * 1531
            mock_llm_instance.embedding.return_value = mock_embedding
            mock_api_llm.return_value = mock_llm_instance
            
            pivot = LLMPivot(online_config)
            text = "Machine learning is a subset of artificial intelligence"
            result = pivot.embedding(text)
            
            print(f"\n[test_embedding_single_text]")
            print(f"Input text: {text}")
            print(f"Embedding vector (first 10): {result[:10]}")
            print(f"Vector dimension: {len(result)}")
            
            mock_llm_instance.embedding.assert_called_once_with(text)
    
    def test_embedding_multiple_texts(self, online_config):
        with patch('llmpivot.llmpivot.ApiLLM') as mock_api_llm:
            mock_llm_instance = Mock()
            mock_embeddings = [
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
            ]
            mock_llm_instance.embedding.return_value = mock_embeddings
            mock_api_llm.return_value = mock_llm_instance
            
            pivot = LLMPivot(online_config)
            texts = ["Artificial intelligence", "Machine learning"]
            result = pivot.embedding(texts)
            
            print(f"\n[test_embedding_multiple_texts]")
            print(f"Input texts: {texts}")
            print(f"Number of embeddings: {len(result)}")
            for i, emb in enumerate(result):
                print(f"Embedding {i+1}: {emb}")
            
            mock_llm_instance.embedding.assert_called_once_with(texts)
    
    def test_perplexity(self, online_config):
        with patch('llmpivot.llmpivot.ApiLLM') as mock_api_llm:
            mock_llm_instance = Mock()
            mock_llm_instance.perplexity.return_value = 18.5
            mock_api_llm.return_value = mock_llm_instance
            
            pivot = LLMPivot(online_config)
            text = "Natural language processing enables computers to understand human language."
            result = pivot.perplexity(text)
            
            print(f"\n[test_perplexity]")
            print(f"Input text: {text}")
            print(f"Perplexity score: {result}")
            
            mock_llm_instance.perplexity.assert_called_once_with(text)
    
    def test_stream_generate(self, online_config):
        with patch('llmpivot.llmpivot.ApiLLM') as mock_api_llm:
            mock_llm_instance = Mock()
            mock_stream = ["Sure", ", ", "I", "'d", " ", "be", " ", "happy", " ", "to", " ", "help", "!"]
            mock_llm_instance.stream_generate.return_value = iter(mock_stream)
            mock_api_llm.return_value = mock_llm_instance
            
            pivot = LLMPivot(online_config)
            messages = [{"role": "user", "content": "Can you help me?"}]
            result = list(pivot.stream_generate(messages))
            
            print(f"\n[test_stream_generate]")
            print(f"Input messages: {messages}")
            print(f"Stream chunks ({len(result)} chunks): {result}")
            print(f"Full response: {''.join(result)}")
            
            mock_llm_instance.stream_generate.assert_called_once_with(messages, None)
    
    def test_cleanup(self, online_config):
        with patch('llmpivot.llmpivot.ApiLLM') as mock_api_llm:
            mock_llm_instance = Mock()
            mock_api_llm.return_value = mock_llm_instance
            
            pivot = LLMPivot(online_config)
            pivot.cleanup()
            
            print(f"\n[test_cleanup]")
            print(f"Cleanup executed successfully")
            
            mock_llm_instance.cleanup.assert_called_once()
    
    def test_context_manager(self, online_config):
        with patch('llmpivot.llmpivot.ApiLLM') as mock_api_llm:
            mock_llm_instance = Mock()
            mock_api_llm.return_value = mock_llm_instance
            
            print(f"\n[test_context_manager]")
            
            with LLMPivot(online_config) as pivot:
                print(f"Pivot instance: {pivot}")
                print(f"Model: {pivot.config.model_id}")
                assert pivot is not None
            
            print(f"Context exited, cleanup invoked")
            mock_llm_instance.cleanup.assert_called_once()
    
    def test_repr(self, online_config):
        with patch('llmpivot.llmpivot.ApiLLM'):
            pivot = LLMPivot(online_config)
            repr_str = repr(pivot)
            
            print(f"\n[test_repr]")
            print(f"String representation: {repr_str}")
            
            assert "online" in repr_str
