import pytest
from unittest.mock import Mock, patch
from llmpivot import LLMPivot
from llmpivot.config import PivotConfig


class TestLocalLLM:
    
    def test_init_with_local_model(self, local_config):
        with patch('llmpivot.llmpivot.LocalLLM') as mock_local_llm:
            pivot = LLMPivot(local_config)
            
            print(f"\n[test_init_with_local_model]")
            print(f"Config: {pivot.config}")
            print(f"Model Type: {pivot.config.model_type}")
            print(f"Model ID: {pivot.config.model_id}")
            
            assert pivot.config == local_config
            mock_local_llm.assert_called_once_with(local_config)
    
    def test_generate(self, local_config):
        with patch('llmpivot.llmpivot.LocalLLM') as mock_local_llm:
            mock_llm_instance = Mock()
            mock_llm_instance.generate.return_value = {
                "content": "Hello! How can I help you today?",
                "role": "assistant",
                "usage": {"total_tokens": 15}
            }
            mock_local_llm.return_value = mock_llm_instance
            
            pivot = LLMPivot(local_config)
            messages = [{"role": "user", "content": "Hello"}]
            result = pivot.generate(messages)
            
            print(f"\n[test_generate]")
            print(f"Input messages: {messages}")
            print(f"Response: {result}")
            
            mock_llm_instance.generate.assert_called_once_with(messages, None)
    
    def test_generate_with_tools(self, local_config, sample_tools):
        with patch('llmpivot.llmpivot.LocalLLM') as mock_local_llm:
            mock_llm_instance = Mock()
            mock_llm_instance.generate.return_value = {
                "content": None,
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "name": "get_weather",
                        "arguments": '{"city": "Beijing"}'
                    }
                ]
            }
            mock_local_llm.return_value = mock_llm_instance
            
            pivot = LLMPivot(local_config)
            messages = [{"role": "user", "content": "What's the weather in Beijing?"}]
            result = pivot.generate(messages, tools=sample_tools)
            
            print(f"\n[test_generate_with_tools]")
            print(f"Input messages: {messages}")
            print(f"Tools: {sample_tools}")
            print(f"Response: {result}")
            
            mock_llm_instance.generate.assert_called_once_with(messages, sample_tools)
    
    def test_dialogue(self, local_config, sample_messages):
        with patch('llmpivot.llmpivot.LocalLLM') as mock_local_llm:
            mock_llm_instance = Mock()
            mock_llm_instance.dialogue.return_value = "Hello! I'm doing well, thank you for asking. How can I assist you today?"
            mock_local_llm.return_value = mock_llm_instance
            
            pivot = LLMPivot(local_config)
            result = pivot.dialogue(sample_messages)
            
            print(f"\n[test_dialogue]")
            print(f"Input messages: {sample_messages}")
            print(f"Response: {result}")
            
            mock_llm_instance.dialogue.assert_called_once_with(sample_messages)
    
    def test_call_function(self, local_config, sample_tools):
        with patch('llmpivot.llmpivot.LocalLLM') as mock_local_llm:
            mock_llm_instance = Mock()
            mock_llm_instance.call_function.return_value = [
                {
                    "id": "call_456",
                    "name": "get_weather",
                    "arguments": '{"city": "Beijing", "unit": "celsius"}'
                }
            ]
            mock_local_llm.return_value = mock_llm_instance
            
            pivot = LLMPivot(local_config)
            messages = [{"role": "user", "content": "Check the weather in Beijing"}]
            result = pivot.call_function(messages, sample_tools)
            
            print(f"\n[test_call_function]")
            print(f"Input messages: {messages}")
            print(f"Tools: {sample_tools}")
            print(f"Function calls: {result}")
            
            mock_llm_instance.call_function.assert_called_once_with(messages, sample_tools)
    
    def test_embedding_single_text(self, local_config):
        with patch('llmpivot.llmpivot.LocalLLM') as mock_local_llm:
            mock_llm_instance = Mock()
            mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
            mock_llm_instance.embedding.return_value = mock_embedding
            mock_local_llm.return_value = mock_llm_instance
            
            pivot = LLMPivot(local_config)
            text = "This is a test sentence"
            result = pivot.embedding(text)
            
            print(f"\n[test_embedding_single_text]")
            print(f"Input text: {text}")
            print(f"Embedding vector (first 5): {result[:5]}")
            print(f"Vector dimension: {len(result)}")
            
            mock_llm_instance.embedding.assert_called_once_with(text)
    
    def test_embedding_multiple_texts(self, local_config):
        with patch('llmpivot.llmpivot.LocalLLM') as mock_local_llm:
            mock_llm_instance = Mock()
            mock_embeddings = [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9]
            ]
            mock_llm_instance.embedding.return_value = mock_embeddings
            mock_local_llm.return_value = mock_llm_instance
            
            pivot = LLMPivot(local_config)
            texts = ["First sentence", "Second sentence", "Third sentence"]
            result = pivot.embedding(texts)
            
            print(f"\n[test_embedding_multiple_texts]")
            print(f"Input texts: {texts}")
            print(f"Number of embeddings: {len(result)}")
            print(f"Embeddings: {result}")
            
            mock_llm_instance.embedding.assert_called_once_with(texts)
    
    def test_perplexity(self, local_config):
        with patch('llmpivot.llmpivot.LocalLLM') as mock_local_llm:
            mock_llm_instance = Mock()
            mock_llm_instance.perplexity.return_value = 25.7
            mock_local_llm.return_value = mock_llm_instance
            
            pivot = LLMPivot(local_config)
            text = "The quick brown fox jumps over the lazy dog."
            result = pivot.perplexity(text)
            
            print(f"\n[test_perplexity]")
            print(f"Input text: {text}")
            print(f"Perplexity: {result}")
            
            mock_llm_instance.perplexity.assert_called_once_with(text)
    
    def test_stream_generate(self, local_config):
        with patch('llmpivot.llmpivot.LocalLLM') as mock_local_llm:
            mock_llm_instance = Mock()
            mock_stream = ["Hello", ", ", "how", " ", "can", " ", "I", " ", "help", "?"]
            mock_llm_instance.stream_generate.return_value = iter(mock_stream)
            mock_local_llm.return_value = mock_llm_instance
            
            pivot = LLMPivot(local_config)
            messages = [{"role": "user", "content": "Hi there"}]
            result = list(pivot.stream_generate(messages))
            
            print(f"\n[test_stream_generate]")
            print(f"Input messages: {messages}")
            print(f"Stream chunks: {result}")
            print(f"Full response: {''.join(result)}")
            
            mock_llm_instance.stream_generate.assert_called_once_with(messages, None)
    
    def test_cleanup(self, local_config):
        with patch('llmpivot.llmpivot.LocalLLM') as mock_local_llm:
            mock_llm_instance = Mock()
            mock_local_llm.return_value = mock_llm_instance
            
            pivot = LLMPivot(local_config)
            pivot.cleanup()
            
            print(f"\n[test_cleanup]")
            print(f"Cleanup called successfully")
            
            mock_llm_instance.cleanup.assert_called_once()
    
    def test_context_manager(self, local_config):
        with patch('llmpivot.llmpivot.LocalLLM') as mock_local_llm:
            mock_llm_instance = Mock()
            mock_local_llm.return_value = mock_llm_instance
            
            print(f"\n[test_context_manager]")
            
            with LLMPivot(local_config) as pivot:
                print(f"Pivot instance created: {pivot}")
                assert pivot is not None
            
            print(f"Context manager exited, cleanup should be called")
            mock_llm_instance.cleanup.assert_called_once()
    
    def test_repr(self, local_config):
        with patch('llmpivot.llmpivot.LocalLLM'):
            pivot = LLMPivot(local_config)
            repr_str = repr(pivot)
            
            print(f"\n[test_repr]")
            print(f"Representation: {repr_str}")
            
            assert "local" in repr_str
