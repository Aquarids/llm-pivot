# test_api_rerank_integration.py
import os
import configparser
from llmpivot import RerankPivot, RerankConfig
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
def api_rerank_config(test_config):
    if not test_config.has_section('online_rerank'):
        pytest.skip("No online_rerank configuration found")

    return RerankConfig(
        model_type="online",
        model_id=test_config.get('online_rerank', 'model_id'),
        api_key=test_config.get('online_rerank', 'api_key'),
        base_url=test_config.get('online_rerank', 'base_url'),
        enable_log=test_config.getboolean('online_rerank', 'enable_log', fallback=False),
    )


@pytest.fixture
def sample_documents():
    return [
        "Python is a high-level programming language.",
        "The Eiffel Tower is located in Paris.",
        "Machine learning is a subset of artificial intelligence.",
        "The Great Wall of China is a famous landmark.",
        "Deep learning uses neural networks with many layers.",
    ]


class TestAPIRerankIntegration:

    @pytest.mark.integration
    @pytest.mark.api
    def test_rerank_basic(self, api_rerank_config, sample_documents):
        with RerankPivot(api_rerank_config) as rerank:
            query = "What is machine learning?"
            results = rerank.rerank_documents(query, sample_documents)
            print(results)

            assert results is not None
            assert isinstance(results, list)
            assert len(results) > 0
            assert all('index' in r and 'relevance_score' in r for r in results)

    @pytest.mark.integration
    @pytest.mark.api
    def test_rerank_order(self, api_rerank_config, sample_documents):
        with RerankPivot(api_rerank_config) as rerank:
            query = "deep learning and neural networks"
            results = rerank.rerank_documents(query, sample_documents)

            scores = [r['relevance_score'] for r in results]
            assert scores == sorted(scores, reverse=True)

    @pytest.mark.integration
    @pytest.mark.api
    def test_rerank_top_n(self, api_rerank_config, sample_documents):
        with RerankPivot(api_rerank_config) as rerank:
            query = "programming language"
            results = rerank.rerank_documents(query, sample_documents, top_n=2)
            print(results)

            assert len(results) <= 2

    @pytest.mark.integration
    @pytest.mark.api
    def test_similarity(self, api_rerank_config):
        with RerankPivot(api_rerank_config) as rerank:
            source = "Machine learning is part of AI."
            sentences = [
                "AI includes machine learning.",
                "The sky is blue.",
                "Deep learning is a type of machine learning.",
            ]
            scores = rerank.similarity(source, sentences)
            print(scores)

            assert scores is not None
            assert isinstance(scores, list)
            assert len(scores) == len(sentences)
            assert all(isinstance(s, float) for s in scores)

    @pytest.mark.integration
    @pytest.mark.api
    def test_context_manager(self, api_rerank_config, sample_documents):
        with RerankPivot(api_rerank_config) as rerank:
            assert rerank is not None
            results = rerank.rerank_documents("test query", sample_documents)
            assert results is not None
