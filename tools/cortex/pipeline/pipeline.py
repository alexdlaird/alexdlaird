__copyright__ = "Copyright (c) 2026 Alex Laird"
__license__ = "MIT"

import json
import re
from pathlib import Path
from typing import Generator, Iterator, List, Union

import requests
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Pipeline is the chat-facing path (Open WebUI). It uses a Q&A-oriented system prompt,
# not the tool-aware agent prompt used by pi/opencode. dev-init-ai copies the prompts/
# directory into the pipelines container alongside this file.
_PROMPT_PATH = Path(__file__).parent / "prompts" / "chat_system_prompt.txt"
CHAT_SYSTEM_PROMPT = _PROMPT_PATH.read_text().strip()


class Pipeline:
    class Valves(BaseModel):
        OLLAMA_BASE_URL: str = "http://host.docker.internal:11434"
        OLLAMA_MODEL: str = "gemma4"
        QDRANT_URL: str = "http://qdrant:6333"
        EMBED_MODEL: str = "nomic-embed-text"
        COLLECTION_NAME: str = "cortex"
        TOP_K: int = 5
        INCLUDE_TESTS: bool = False
        SYSTEM_PROMPT: str = CHAT_SYSTEM_PROMPT

    def __init__(self):
        self.id = "cortex"
        self.name = "cortex"
        self.valves = self.Valves()
        self._resolved_model = None

    async def on_startup(self):
        pass

    async def on_valves_updated(self):
        self._resolved_model = None

    def _resolve_model(self) -> str:
        if self._resolved_model:
            return self._resolved_model
        try:
            response = requests.get(f"{self.valves.OLLAMA_BASE_URL}/api/tags", timeout=5)
            models = [m["name"] for m in response.json().get("models", [])]
            # Exact "cortex" or "cortex:TAG" — exclude derived names like "cortex-agent"
            # which are the tool-calling overlay, not the chat-facing model.
            for m in models:
                base = m.split(":")[0]
                if base == "cortex":
                    self._resolved_model = m
                    return self._resolved_model
        except Exception:
            pass
        self._resolved_model = self.valves.OLLAMA_MODEL
        return self._resolved_model

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
            node = json.loads(node_content) if isinstance(node_content, str) else node_content
            text = node.get("text", "").strip()
            file_path = point.payload.get("file_path") or node.get("metadata", {}).get("file_path", "unknown")
            if text:
                chunks.append(f"// {file_path}\n{text}")

        return "\n\n---\n\n".join(chunks)

    def _iter_stream(self, response, forward_reasoning: bool) -> Generator:
        """
        Consume Ollama SSE. Reasoning tokens are in delta.reasoning; response tokens
        in delta.content. For stream clients (forward_reasoning=True), wrap reasoning
        in <think>…</think> so Open WebUI renders it as a collapsible section.
        """
        reasoning_open = False

        for line in response.iter_lines():
            if not line:
                continue
            raw = line.decode("utf-8")
            if not raw.startswith("data: "):
                continue
            data_str = raw[6:]
            if data_str.strip() == "[DONE]":
                break
            try:
                data = json.loads(data_str)
            except Exception:
                continue
            choices = data.get("choices") or [{}]
            delta = choices[0].get("delta", {})
            reasoning = delta.get("reasoning") or ""
            content = delta.get("content") or ""

            if forward_reasoning and reasoning:
                if not reasoning_open:
                    yield "<think>"
                    reasoning_open = True
                yield reasoning
            if content:
                if reasoning_open:
                    yield "</think>"
                    reasoning_open = False
                yield content

        if reasoning_open:
            yield "</think>"

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
        clean_history = []
        for m in filtered:
            if m.get("role") == "assistant" and m.get("content"):
                m = {**m, "content": _THINK_RE.sub("", m["content"]).strip()}
            clean_history.append(m)
        augmented_messages = [{"role": "system", "content": system_content}] + clean_history

        client_wants_stream = body.get("stream", True)

        try:
            response = requests.post(
                f"{self.valves.OLLAMA_BASE_URL}/v1/chat/completions",
                json={
                    "model": self._resolve_model(),
                    "messages": augmented_messages,
                    "stream": True,
                },
                stream=True,
            )
            response.raise_for_status()
        except Exception as e:
            yield f"Pipeline error contacting Ollama: {e}"
            return

        try:
            if client_wants_stream:
                yield from self._iter_stream(response, forward_reasoning=True)
            else:
                yield "".join(self._iter_stream(response, forward_reasoning=False))
        except Exception as e:
            yield f"Pipeline error streaming response: {e}"
