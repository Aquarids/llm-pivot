import subprocess
import time
import requests
import ollama
from typing import List, Dict, Any, Optional, Iterator
from .logger import Logger


class OllamaHelper:
    
    def __init__(self, logger: Logger, host: str = "http://localhost:11434",):
        self.host = host
        self.process = None
        self._should_cleanup = False
        self.logger = logger
    
    def is_running(self) -> bool:
        try:
            response = requests.get(self.host, timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def start(self, timeout: int = 30):
        if self.is_running():
            self.logger.info(f"Ollama service already running at {self.host}")
            return
        
        self.logger.info("Starting Ollama service...")
        
        try:
            self.process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
        except FileNotFoundError:
            self.logger.error("ollama command not found")
            raise RuntimeError("ollama command not found")
        
        self._should_cleanup = True
        
        for i in range(timeout):
            if self.is_running():
                self.logger.info(f"Ollama service started successfully in {i+1}s")
                return
            time.sleep(1)
        
        if self.process:
            self.process.kill()
        
        self.logger.error(f"Ollama start timeout ({timeout}s)")
        raise RuntimeError(f"Ollama start timeout ({timeout}s)")
    
    def stop(self):
        if self.process and self._should_cleanup:
            self.logger.info("Stopping Ollama service...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
                self.logger.info("Ollama service stopped")
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.logger.warning("Ollama service force killed")
            self.process = None
            self._should_cleanup = False
    
    def ensure_model(self, model: str):
        self.logger.debug(f"Checking model: {model}")
        try:
            ollama.show(model)
            self.logger.info(f"Model {model} is ready")
        except:
            self.logger.info(f"Pulling model: {model}")
            try:
                ollama.pull(model)
                self.logger.info(f"Model {model} pulled successfully")
            except Exception as e:
                self.logger.error(f"Failed to pull model {model}: {str(e)}")
                raise
    
    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs
    ):
        self.logger.debug(f"Chat request to model {model}, stream={stream}")
        params = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        
        if "options" in kwargs:
            params["options"] = kwargs["options"]
        
        if tools:
            params["tools"] = tools
            self.logger.debug(f"Using {len(tools)} tools")
        
        return ollama.chat(**params)
    
    def generate(self, model: str, prompt: str, **kwargs):
        self.logger.debug(f"Generate request to model {model}")
        return ollama.generate(model=model, prompt=prompt, **kwargs)
    
    def embeddings(self, model: str, prompt: str):
        self.logger.debug(f"Embedding request to model {model}")
        return ollama.embeddings(model=model, prompt=prompt)
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
