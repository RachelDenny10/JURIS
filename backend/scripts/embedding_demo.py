"""
Offline test of the embedding service (no server needed).

The sentences below are DEMO TEXT written to test semantic similarity;
they are not authentic legal documents.

Run from the backend folder:
    python scripts\\embedding_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from app.services import embedding_service  # noqa: E402

SENTENCES = [
    "A consumer can seek a refund if an online seller refuses to return the money.",
    "The vendor denied reimbursement for the purchase made on the website.",
    "The District Commission hears complaints where the value of goods is below the limit.",
    "Misleading advertisements are prohibited under the guidelines.",
]
QUERY = "What can I do if a shop will not give my money back?"


def main() -> None:
    info = embedding_service.get_model_info()
    print(f"Model: {info['model_name']}   dimension: {info['dimension']}\n")

    vectors = embedding_service.embed_texts(SENTENCES)
    query_vector = embedding_service.embed_query(QUERY)

    norms = np.linalg.norm(vectors, axis=1)
    print("Vector norms (should all be ~1.0):", np.round(norms, 3).tolist(), "\n")

    scores = vectors @ query_vector
    print(f"Query: {QUERY}\n")
    for index in np.argsort(scores)[::-1]:
        print(f"  {scores[index]:.3f}  {SENTENCES[index]}")

    print("\nExpected: the two refund sentences (1 and 2) should score highest,")
    print("even though they share almost no words with the query.")

    sample = (
        "First paragraph about refunds. " * 30
        + "\n\nSecond paragraph about complaints. " * 30
        + "\n----- PAGE BREAK -----\n"
        + "Third paragraph on the next page. " * 30
    )
    chunks = embedding_service.chunk_text(sample, target_words=60, overlap_words=10)
    print(f"\nChunking test: {len(chunks)} chunks")
    for chunk in chunks:
        print(f"  chunk {chunk.chunk_id}: {chunk.word_count} words, pages {chunk.page_start}-{chunk.page_end}")


if __name__ == "__main__":
    main()