import os
os.environ['RUST_LOG'] = 'error,reqwest=off,hyper=off,h2=off'

import configparser
from pathlib import Path
import sys
from llmpivot import LLMPivot, PivotConfig

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))



def load_test_config():
    config = configparser.ConfigParser()
    config_path = os.path.join(project_root, 'tests', 'local_config.ini')
    
    if os.path.exists(config_path):
        config.read(config_path, encoding='utf-8')
    
    return config

test_config = load_test_config()

llm = LLMPivot(PivotConfig(
    model_type="online",
    api_type="responses",
    model_id=test_config['online']['model_id'],
    api_key=test_config['online']['api_key'],
    base_url=test_config['online']['base_url'],
    llm_default_params={
        "temperature": 1.0,
        "extra_body": {"thinking": {"type": "disabled"}}
    }
))


messages = [{"role": "user", "content": "Who r u?"}]

response = llm.dialogue(messages, stream=True)
print(response)