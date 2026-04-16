__copyright__ = "Copyright (c) 2026 Alex Laird"
__license__ = "MIT"

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from config import COLLECTION_NAME, EMBED_MODEL, OLLAMA_BASE_URL, QDRANT_URL


def search(query: str, top_k: int, include_tests: bool = False) -> None:
    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    Settings.llm = None

    client = QdrantClient(url=QDRANT_URL)
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
    index = VectorStoreIndex.from_vector_store(vector_store)

    filters = None if include_tests else MetadataFilters(
        filters=[MetadataFilter(key="chunk_type", value="source")]
    )
    retriever = index.as_retriever(similarity_top_k=top_k, filters=filters)

    nodes = retriever.retrieve(query)

    for i, node in enumerate(nodes, 1):
        source = node.metadata.get("file_path", "unknown")
        score = f"{node.score:.3f}" if node.score is not None else "n/a"
        print(f"\n[{i}] {source} (score: {score})")
        print("-" * 60)
        print(node.text.strip())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search the RAG index.")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("--include-tests", action="store_true", help="Include test files in results")
    args = parser.parse_args()

    search(args.query, args.top_k, include_tests=args.include_tests)
