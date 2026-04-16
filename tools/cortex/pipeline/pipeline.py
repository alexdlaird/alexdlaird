__copyright__ = "Copyright (c) 2026 Alex Laird"
__license__ = "MIT"

from typing import Generator, Iterator, List, Union

import requests
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue


class Pipeline:
    class Valves(BaseModel):
        OLLAMA_BASE_URL: str = "http://host.docker.internal:11434"
        OLLAMA_MODEL: str = "cortex"
        QDRANT_URL: str = "http://qdrant:6333"
        EMBED_MODEL: str = "nomic-embed-text"
        COLLECTION_NAME: str = "cortex"
        TOP_K: int = 5
        INCLUDE_TESTS: bool = False
        SYSTEM_PROMPT: str = (
            "You are a coding assistant with deep knowledge of Alex Laird's personal projects "
            "including HeliumEdu (Django + Flutter), pyngrok and java-ngrok, and amazon-orders. "
            "You follow his coding conventions and architectural patterns. Provide direct, "
            "technical answers."
        )

    def __init__(self):
        self.name = "Cortex RAG Pipeline"
        self.valves = self.Valves()

    async def on_startup(self):
        pass

    async def on_valves_updated(self):
        pass

    def _embed(self, text: str) -> list:
        response = requests.post(
            f"{self.valves.OLLAMA_BASE_URL}/api/embed",
            json={"model": self.valves.EMBED_MODEL, "input": text},
        )
        response.raise_for_status()
        return response.json()["embeddings"][0]

    def _retrieve_context(self, query: str) -> str:
        vector = self._embed(query)

        client = QdrantClient(url=self.valves.QDRANT_URL)
        query_filter = None if self.valves.INCLUDE_TESTS else Filter(
            must=[FieldCondition(key="chunk_type", match=MatchValue(value="source"))]
        )

        results = client.query_points(
            collection_name=self.valves.COLLECTION_NAME,
            query=vector,
            query_filter=query_filter,
            limit=self.valves.TOP_K,
            with_payload=True,
        )

        chunks = []
        for point in results.points:
            node_content = point.payload.get("_node_content", "{}")
            import json
            node = json.loads(node_content) if isinstance(node_content, str) else node_content
            text = node.get("text", "").strip()
            file_path = point.payload.get("file_path") or node.get("metadata", {}).get("file_path", "unknown")
            if text:
                chunks.append(f"// {file_path}\n{text}")

        return "\n\n---\n\n".join(chunks)

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict,
    ) -> Union[str, Generator, Iterator]:
        try:
            context = self._retrieve_context(user_message)
        except Exception as e:
            print(f"Warning: RAG retrieval failed ({e}), proceeding without context")
            context = ""

        system_content = self.valves.SYSTEM_PROMPT
        if context:
            system_content += f"\n\nUse the following code excerpts to inform your answer:\n\n{context}"

        filtered = [m for m in messages if m.get("role") != "system"]
        augmented_messages = [{"role": "system", "content": system_content}] + filtered

        stream = body.get("stream", True)

        response = requests.post(
            f"{self.valves.OLLAMA_BASE_URL}/v1/chat/completions",
            json={
                "model": self.valves.OLLAMA_MODEL,
                "messages": augmented_messages,
                "stream": stream,
            },
            stream=stream,
        )
        response.raise_for_status()

        if stream:
            for line in response.iter_lines():
                if line:
                    yield line.decode("utf-8") + "\n"
        else:
            yield response.text
