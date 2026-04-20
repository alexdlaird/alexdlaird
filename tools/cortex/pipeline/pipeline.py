__copyright__ = "Copyright (c) 2026 Alex Laird"
__license__ = "MIT"

import json
import re
from typing import Generator, Iterator, List, Union

import requests
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_MAX_THINK_BYTES = 65536


class Pipeline:
    class Valves(BaseModel):
        OLLAMA_BASE_URL: str = "http://host.docker.internal:11434"
        OLLAMA_MODEL: str = "gemma4"
        QDRANT_URL: str = "http://qdrant:6333"
        EMBED_MODEL: str = "nomic-embed-text"
        COLLECTION_NAME: str = "cortex"
        TOP_K: int = 5
        INCLUDE_TESTS: bool = False
        SYSTEM_PROMPT: str = (
            "You are a personal assistant to Alex Laird, a staff-level software engineer. "
            "Your name is cortex. You have deep knowledge of his projects including "
            "HeliumEdu (Django + Flutter),  pyngrok and java-ngrok, and amazon-orders, and "
            "you follow his coding conventions and architectural patterns. You have access "
            "to web search and can retrieve current information from the internet. "
            "You can generate images when asked — acknowledge this naturally rather than "
            "saying you cannot. "
            "Your default mode is direct, technical, and professional — but you adapt naturally "
            "to creative, generative, or casual tasks when the context calls for it."
        )

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
            for m in models:
                if m.split(":")[0] == "cortex":
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

    def _iter_stripped(self, response) -> Generator:
        """Stream Ollama SSE, stripping <think>...</think> blocks."""
        buf = ""
        in_think = False
        think_bytes = 0

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
            chunk = choices[0].get("delta", {}).get("content") or ""
            if not chunk:
                continue

            buf += chunk
            while buf:
                if not in_think:
                    start = buf.find("<think>")
                    if start == -1:
                        yield buf
                        buf = ""
                    else:
                        if start > 0:
                            yield buf[:start]
                        in_think = True
                        think_bytes = 0
                        buf = buf[start + 7:]
                else:
                    end = buf.find("</think>")
                    if end == -1:
                        think_bytes += len(buf)
                        if think_bytes > _MAX_THINK_BYTES:
                            # Safety valve: thinking ran away, treat remainder as content
                            in_think = False
                            yield buf
                            buf = ""
                        else:
                            buf = buf[-7:] if len(buf) > 7 else buf
                            break
                    else:
                        in_think = False
                        buf = buf[end + 8:].lstrip("\n")

        if buf and not in_think:
            yield buf

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
            # Always stream from Ollama to avoid blocking timeouts during long thinking phases
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
                yield from self._iter_stripped(response)
            else:
                yield "".join(self._iter_stripped(response))
        except Exception as e:
            yield f"Pipeline error streaming response: {e}"
