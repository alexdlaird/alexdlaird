__copyright__ = "Copyright (c) 2026 Alex Laird"
__license__ = "MIT"

import argparse
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from config import (
    COLLECTION_NAME,
    CODE_EXTENSIONS,
    DEVELOPER_DIR,
    DOC_EXTENSIONS,
    EMBED_MODEL,
    OLLAMA_BASE_URL,
    QDRANT_URL,
    REPOS,
)

logger = logging.getLogger(__name__)

# Suppress noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("llama_index").setLevel(logging.ERROR)


def build_client_and_store(rebuild: bool) -> tuple[QdrantClient, QdrantVectorStore]:
    client = QdrantClient(url=QDRANT_URL)

    if rebuild and client.collection_exists(COLLECTION_NAME):
        logger.info(f"Deleting existing collection: {COLLECTION_NAME}")
        client.delete_collection(COLLECTION_NAME)
        logger.info("Collection deleted.")

    if not client.collection_exists(COLLECTION_NAME):
        logger.info(f"Creating collection: {COLLECTION_NAME}")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

    return client, QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)


TEST_PATH_PATTERN = re.compile(r"(^|/)tests?(/|$)", re.IGNORECASE)


def is_test_file(path: str) -> bool:
    return bool(TEST_PATH_PATTERN.search(path))


def load_repo(path: Path, extensions: list[str]):
    try:
        return SimpleDirectoryReader(
            input_dir=str(path),
            recursive=True,
            required_exts=extensions,
            exclude=["**/__pycache__/**", "**/.git/**", "**/node_modules/**", "**/.venv/**", "**/venv/**", "**/build/**"],
        ).load_data()
    except Exception:
        logger.warning(f"No matching files found in {path}, skipping.")
        return []


def ingest_repos(repos: list[str], rebuild: bool) -> None:
    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    Settings.llm = None

    client, vector_store = build_client_and_store(rebuild)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=64)

    for repo in repos:
        repo_path = DEVELOPER_DIR / repo
        if not repo_path.exists():
            logger.warning(f"Repo path not found, skipping: {repo_path}")
            continue

        logger.info(f"Ingesting: {repo_path}")

        docs = load_repo(repo_path, CODE_EXTENSIONS + DOC_EXTENSIONS)
        if docs:
            for doc in docs:
                file_path = doc.metadata.get("file_path", "")
                doc.metadata["chunk_type"] = "test" if is_test_file(file_path) else "source"
            logger.info(f"  Found {len(docs)} files, embedding ...")
            VectorStoreIndex.from_documents(
                docs,
                storage_context=storage_context,
                transformations=[splitter],
            )
            logger.info(f"  Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Ingest repos into the RAG index.")
    parser.add_argument("--rebuild", action="store_true", help="Drop and rebuild the collection from scratch.")
    args = parser.parse_args()

    ingest_repos(REPOS, rebuild=args.rebuild)
