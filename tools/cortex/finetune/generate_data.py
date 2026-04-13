__copyright__ = "Copyright (c) 2026 Alex Laird"
__license__ = "MIT"

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import requests
from qdrant_client import QdrantClient

from config import COLLECTION_NAME, FINETUNE_DATA_DIR, OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL, QDRANT_URL

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a coding assistant. Given a code or documentation chunk, generate ONE clear "
    "technical question a developer might ask about it, and a thorough answer. "
    'Respond ONLY with valid JSON: {"question": "...", "answer": "..."}'
)


def scroll_chunks(client, collection, sample_every):
    offset = None
    index = 0
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            if index % sample_every == 0:
                node_data = json.loads(point.payload.get("_node_content", "{}"))
                text = node_data.get("text", "").strip()
                if text:
                    yield text
            index += 1
        if offset is None:
            break


def generate_pair(chunk_text, ollama_base_url, model):
    try:
        response = requests.post(
            f"{ollama_base_url}/api/chat",
            json={
                "model": model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": chunk_text},
                ],
            },
            timeout=60,
        )
        response.raise_for_status()
        content = response.json()["message"]["content"].strip()
        content = content.removeprefix("```json").removesuffix("```").strip()
        pair = json.loads(content)
        question = pair.get("question", "").strip()
        answer = pair.get("answer", "").strip()
        if question and answer:
            return question, answer
    except Exception as e:
        logger.warning(f"Skipping chunk — generation failed: {e}")
    return None


def to_sharegpt(question, answer):
    return {"conversations": [{"from": "human", "value": question}, {"from": "gpt", "value": answer}]}


def generate_data(collection, limit, sample_every, output_path, ollama_base_url, model, append):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"

    seen_questions = set()
    if append and output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    record = json.loads(line)
                    seen_questions.add(record["conversations"][0]["value"])
                except Exception:
                    pass

    client = QdrantClient(url=QDRANT_URL)
    total, skipped, written = 0, 0, 0

    with open(output_path, mode) as out:
        for chunk_text in scroll_chunks(client, collection, sample_every):
            if limit and total >= limit:
                break
            total += 1

            result = generate_pair(chunk_text, ollama_base_url, model)
            if result is None:
                skipped += 1
                continue

            question, answer = result
            if question in seen_questions:
                skipped += 1
                continue

            seen_questions.add(question)
            out.write(json.dumps(to_sharegpt(question, answer)) + "\n")
            written += 1

            if written % 50 == 0:
                logger.info(f"  {written} pairs written ({skipped} skipped) ...")

    logger.info(f"Done — {written} pairs written to {output_path} ({skipped} skipped out of {total} chunks)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Generate ShareGPT training data from RAG corpus.")
    parser.add_argument("--limit", type=int, default=0, help="Max chunks to process (0 = all)")
    parser.add_argument("--sample-every", type=int, default=1, help="Take every Nth chunk (default: 1 = all)")
    parser.add_argument("--output", type=Path, default=None, help="Output JSONL path")
    parser.add_argument("--collection", default=COLLECTION_NAME, help="Qdrant collection name")
    parser.add_argument("--ollama-model", default=OLLAMA_CHAT_MODEL, help="Ollama model for generation")
    parser.add_argument("--append", action="store_true", help="Append to existing output instead of overwriting")
    args = parser.parse_args()

    output_path = args.output or (FINETUNE_DATA_DIR / "sharegpt.jsonl")

    generate_data(
        collection=args.collection,
        limit=args.limit,
        sample_every=args.sample_every,
        output_path=output_path,
        ollama_base_url=OLLAMA_BASE_URL,
        model=args.ollama_model,
        append=args.append,
    )
