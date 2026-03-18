# infinity_helper.py
import subprocess
import time
import httpx
from typing import List, Dict, Any, Optional

from .logger import Logger


class InfinityHelper:

    def __init__(
        self,
        logger: Logger,
        host: str = "http://localhost:7997",
        model_id: Optional[str] = None,
        device: str = "cuda",
        extra_args: Optional[List[str]] = None,
    ):
        self.logger = logger
        self.host = host.rstrip("/")
        self.model_id = model_id
        self.device = device
        self.extra_args = extra_args or []
        self._process: Optional[subprocess.Popen] = None
        self._client = httpx.Client(base_url=self.host, timeout=60.0)

    def start(self):
        if self._process and self._process.poll() is None:
            self.logger.info("Infinity server is already running")
            return

        if not self.model_id:
            raise ValueError("model_id is required to start Infinity server")

        port = self._parse_port()

        cmd = [
            "infinity_emb", "v2",
            "--model-id", self.model_id,
            "--port", str(port),
            "--device", self.device,
        ] + self.extra_args

        self.logger.info(f"Starting Infinity server: {' '.join(cmd)}")

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self._wait_until_ready()

    def _parse_port(self) -> int:
        try:
            from urllib.parse import urlparse
            parsed = urlparse(self.host)
            return parsed.port or 7997
        except Exception:
            return 7997

    def _wait_until_ready(self, timeout: int = 60, interval: float = 2.0):
        self.logger.info("Waiting for Infinity server to be ready...")
        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = self._client.get("/health")
                if resp.status_code == 200:
                    self.logger.info("Infinity server is ready")
                    return
            except Exception:
                pass
            if self._process.poll() is not None:
                stderr = self._process.stderr.read().decode()
                raise RuntimeError(f"Infinity server process exited unexpectedly:\n{stderr}")
            time.sleep(interval)
        raise TimeoutError(f"Infinity server did not become ready within {timeout}s")

    def stop(self):
        if self._process:
            self.logger.info("Stopping Infinity server")
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.logger.warning("Force killing Infinity server")
                self._process.kill()
            self._process = None
        self._client.close()

    def is_running(self) -> bool:
        if not self._process:
            return False
        return self._process.poll() is None

    def embeddings(
        self,
        model: str,
        input: List[str],
        **kwargs,
    ) -> List[List[float]]:
        payload = {"model": model, "input": input, **kwargs}
        self.logger.debug(f"Infinity embeddings request: model={model}, input_count={len(input)}")
        resp = self._client.post("/v1/embeddings", json=payload)
        resp.raise_for_status()
        data = resp.json()
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]

    def rerank(
        self,
        model: str,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        payload = {
            "model": model,
            "query": query,
            "documents": documents,
            **kwargs,
        }
        if top_n is not None:
            payload["top_n"] = top_n

        self.logger.debug(f"Infinity rerank request: model={model}, docs={len(documents)}")
        resp = self._client.post("/v1/rerank", json=payload)
        resp.raise_for_status()
        results = resp.json()["results"]
        sorted_results = sorted(results, key=lambda x: x["index"])
        return [
            {
                "index": item["index"],
                "text": item["document"]["text"],
                "relevance_score": item["relevance_score"],
            }
            for item in sorted_results
        ]

    def similarity(
        self,
        model: str,
        source_sentence: str,
        sentences: List[str],
        normalize: bool = True,
        **kwargs,
    ) -> List[float]:
        payload = {
            "model": model,
            "inputs": {
                "source_sentence": source_sentence,
                "sentences": sentences,
            },
            "normalize": normalize,
            **kwargs,
        }
        self.logger.debug(f"Infinity similarity request: model={model}, sentences={len(sentences)}")
        resp = self._client.post("/v1/sentence-similarity", json=payload)
        resp.raise_for_status()
        return resp.json()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
