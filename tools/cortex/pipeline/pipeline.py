__copyright__ = "Copyright (c) 2026 Alex Laird"
__license__ = "MIT"

from typing import Generator, Iterator, List, Optional, Union

import requests
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from pydantic import BaseModel
from qdrant_client import QdrantClient


class Pipeline:
    class Valves(BaseModel):
        OLLAMA_BASE_URL: str = "http://host.docker.internal:11434"
        OLLAMA_MODEL: str = "cortex"
        QDRANT_URL: str = "http://host.docker.internal:6333"
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
        self._retriever = None

    async def on_startup(self):
        self._init_retriever()

    async def on_valves_updated(self):
        self._init_retriever()

    def _init_retriever(self):
        Settings.embed_model = OllamaEmbedding(
            model_name=self.valves.EMBED_MODEL,
            base_url=self.valves.OLLAMA_BASE_URL,
        )
        Settings.llm = None

        client = QdrantClient(url=self.valves.QDRANT_URL)
        vector_store = QdrantVectorStore(client=client, collection_name=self.valves.COLLECTION_NAME)
        index = VectorStoreIndex.from_vector_store(vector_store)

        filters = None if self.valves.INCLUDE_TESTS else MetadataFilters(
            filters=[MetadataFilter(key="chunk_type", value="source")]
        )
        self._retriever = index.as_retriever(similarity_top_k=self.valves.TOP_K, filters=filters)

    def _retrieve_context(self, query: str) -> str:
        if self._retriever is None:
            self._init_retriever()

        nodes = self._retriever.retrieve(query)
        if not nodes:
            return ""

        chunks = []
        for node in nodes:
            source = node.metadata.get("file_path", "unknown")
            chunks.append(f"// {source}\n{node.text.strip()}")

        return "\n\n---\n\n".join(chunks)

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict,
    ) -> Union[str, Generator, Iterator]:
        context = self._retrieve_context(user_message)

        system_content = self.valves.SYSTEM_PROMPT
        if context:
            system_content += f"\n\nUse the following code excerpts to inform your answer:\n\n{context}"

        filtered = [m for m in messages if m.get("role") != "system"]
        augmented_messages = [{"role": "system", "content": system_content}] + filtered

        response = requests.post(
            f"{self.valves.OLLAMA_BASE_URL}/v1/chat/completions",
            json={
                **body,
                "model": self.valves.OLLAMA_MODEL,
                "messages": augmented_messages,
            },
            stream=body.get("stream", False),
        )
        response.raise_for_status()

        if body.get("stream", False):
            for line in response.iter_lines():
                if line:
                    yield line.decode("utf-8") + "\n"
        else:
            yield response.text
