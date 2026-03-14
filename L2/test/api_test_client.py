from __future__ import annotations

from pathlib import Path

import requests
from sentence_transformers import SentenceTransformer

API_URL = "http://127.0.0.1:8000"
TOP_K = 3
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def upload_all_txt_files(test_dir: Path) -> None:
    txt_files = sorted(test_dir.glob("*.txt"))
    if not txt_files:
        print("No .txt files found in test folder.")
        return

    for file_path in txt_files:
        content = file_path.read_text(encoding="utf-8")
        payload = {
            "document_id": file_path.stem,
            "source": file_path.name,
            "content": content,
            "chunk_size": 700,
            "chunk_overlap": 120,
        }
        response = requests.post(f"{API_URL}/add_document", json=payload, timeout=120)
        if response.ok:
            print(f"[OK] {file_path.name}: {response.json()}")
        else:
            print(f"[ERROR] {file_path.name}: {response.status_code} {response.text}")


def interactive_query_loop(model: SentenceTransformer) -> None:
    print("Type your question and press Enter.")
    print("Type 'exit' to stop.\n")

    while True:
        user_text = input("Query> ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit", "q"}:
            print("Stopping.")
            break

        query_vector = model.encode([user_text], normalize_embeddings=True)[0].tolist()
        payload = {"query_vector": query_vector, "top_k": TOP_K}
        response = requests.post(f"{API_URL}/query", json=payload, timeout=120)

        if not response.ok:
            print(f"[ERROR] {response.status_code}: {response.text}\n")
            continue

        result = response.json()
        rows = result.get("results", [])
        if not rows:
            print("No matches found.\n")
            continue

        print("\nTop matches:")
        for i, item in enumerate(rows, start=1):
            preview = item["content"][:180].replace("\n", " ")
            print(
                f"{i}. doc={item['document_id']} "
                f"chunk={item['chunk_index']} "
                f"distance={item['distance']:.4f}\n"
                f"   {preview}..."
            )
        print()


def main() -> None:
    test_dir = Path(__file__).resolve().parent

    print("Loading embedding model. This may take a while on first run...")
    model = SentenceTransformer(MODEL_NAME)

    upload_all_txt_files(test_dir)
    interactive_query_loop(model)


if __name__ == "__main__":
    main()
