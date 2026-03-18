# test_api_embed_integration.py
import os
import configparser
from llmpivot import EmbedPivot, EmbedConfig
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
def api_embed_config(test_config):
    if not test_config.has_section('online_embed'):
        pytest.skip("No online_embed configuration found")

    return EmbedConfig(
        model_type="online",
        model_id=test_config.get('online_embed', 'model_id'),
        api_key=test_config.get('online_embed', 'api_key'),
        base_url=test_config.get('online_embed', 'base_url'),
        enable_log=test_config.getboolean('online_embed', 'enable_log', fallback=False),
    )


class TestAPIEmbedIntegration:

    @pytest.mark.integration
    @pytest.mark.api
    def test_embedding_single(self, api_embed_config):
        with EmbedPivot(api_embed_config) as embed:
            result = embed.embedding("Hello, world!")
            print(result[:5])

            assert result is not None
            assert isinstance(result, list)
            assert isinstance(result[0], float)
            assert len(result) > 0

    @pytest.mark.integration
    @pytest.mark.api
    def test_embedding_batch(self, api_embed_config):
        with EmbedPivot(api_embed_config) as embed:
            texts = ["Hello", "World", "Foo"]
            result = embed.embedding(texts)
            print([r[:3] for r in result])

            assert result is not None
            assert isinstance(result, list)
            assert len(result) == len(texts)
            assert all(isinstance(r, list) for r in result)
            assert all(isinstance(v, float) for v in result[0])

    @pytest.mark.integration
    @pytest.mark.api
    def test_embedding_consistency(self, api_embed_config):
        with EmbedPivot(api_embed_config) as embed:
            text = "Consistency test"
            result1 = embed.embedding(text)
            result2 = embed.embedding(text)

            assert len(result1) == len(result2)
            assert result1 == result2

    @pytest.mark.integration
    @pytest.mark.api
    def test_embedding_dimension_consistent(self, api_embed_config):
        with EmbedPivot(api_embed_config) as embed:
            texts = ["Short text", "A much longer text that contains more words and information"]
            result = embed.embedding(texts)

            assert len(result[0]) == len(result[1])

    @pytest.mark.integration
    @pytest.mark.api
    def test_context_manager(self, api_embed_config):
        with EmbedPivot(api_embed_config) as embed:
            assert embed is not None
            result = embed.embedding("Hi")
            assert result is not None
