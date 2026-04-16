__copyright__ = "Copyright (c) 2026 Alex Laird"
__license__ = "MIT"

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from config import COLLECTION_NAME, DEVELOPER_DIR, DOC_EXTENSIONS, DOC_PATHS, EMBED_MODEL, OLLAMA_BASE_URL, QDRANT_URL

logger = logging.getLogger(__name__)

# Suppress noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("llama_index").setLevel(logging.WARNING)


def ingest_docs(doc_paths: list[str]) -> None:
    if not doc_paths:
        logger.info("No doc paths configured in config.py, skipping.")
        return

    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    Settings.llm = None

    client = QdrantClient(url=QDRANT_URL)
    vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=64)

    for raw_path in doc_paths:
        path = Path(raw_path) if Path(raw_path).is_absolute() else DEVELOPER_DIR / raw_path
        if not path.exists():
            logger.warning(f"Doc path not found, skipping: {path}")
            continue

        logger.info(f"Ingesting docs: {path}")

        try:
            docs = SimpleDirectoryReader(
                input_dir=str(path) if path.is_dir() else None,
                input_files=[str(path)] if path.is_file() else None,
                recursive=True,
                required_exts=DOC_EXTENSIONS,
            ).load_data()
        except Exception as e:
            logger.warning(f"Error loading {path}: {e}")
            continue

        if docs:
            VectorStoreIndex.from_documents(
                docs,
                storage_context=storage_context,
                transformations=[splitter],
            )
            logger.info(f"  Indexed {len(docs)} docs from {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ingest_docs(DOC_PATHS)
