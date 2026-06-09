import os
os.environ['RUST_LOG'] = 'error,reqwest=off,hyper=off,h2=off'

import configparser
from pathlib import Path
import sys
from llmpivot import EmbedPivot, EmbedConfig

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_test_config():
    config = configparser.ConfigParser()
    config_path = os.path.join(project_root, 'tests', 'local_config.ini')

    if os.path.exists(config_path):
        config.read(config_path, encoding='utf-8')

    return config


test_config = load_test_config()

embed = EmbedPivot(EmbedConfig(
    model_type="online",
    model_id=test_config['online_embed']['model_id'],
    api_key=test_config['online_embed']['api_key'],
    base_url=test_config['online_embed']['base_url'],
    embed_default_params={
        "encoding_format": "float",
    }
))

text = "Who r u?"
vector = embed.embedding(text)
print(f"Single embedding dimension: {len(vector)}")
print(vector[:5])

texts = ["Who r u?", "Hello, world!"]
vectors = embed.embedding(texts)
print(f"Batch embedding count: {len(vectors)}")
print(f"Batch embedding dimension: {len(vectors[0])}")
print([item[:5] for item in vectors])
