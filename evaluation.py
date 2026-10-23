
import json
from pathlib import Path

from dotenv import load_dotenv

from rag_pipeline import (
    DEFAULT_MIN_RELEVANCE,
    DEFAULT_TOP_K,
    get_retrieved_documents,
)

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "test_questions.json"


def precision_at_k(retrieved, relevant, k):
    results = retrieved[:k]

    if not results:
        return 0.0

    return sum(chunk in relevant for chunk in results) / len(results)


def recall_at_k(retrieved, relevant, k):
    if not relevant:
        return 0.0

    return sum(chunk in relevant for chunk in retrieved[:k]) / len(relevant)


def reciprocal_rank(retrieved, relevant):
    for rank, chunk in enumerate(retrieved, start=1):
        if chunk in relevant:
            return 1.0 / rank

    return 0.0


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {DATASET_PATH}"
        )

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    if not dataset:
        raise ValueError("Evaluation dataset is empty.")

    results = []

    for item in dataset:
        relevant = set(item["relevant_chunk_ids"])

        retrieved_docs = get_retrieved_documents(
            question=item["question"],
            top_k=DEFAULT_TOP_K,
            min_relevance=DEFAULT_MIN_RELEVANCE,
            mode="similarity",
        )

        retrieved_ids = [
            doc.metadata.get("chunk_id", "")
            for doc, score in retrieved_docs
        ]

        result = {
            "question": item["question"],
            "retrieved_chunk_ids": retrieved_ids,
            "precision_at_k": precision_at_k(
                retrieved_ids, relevant, DEFAULT_TOP_K
            ),
            "recall_at_k": recall_at_k(
                retrieved_ids, relevant, DEFAULT_TOP_K
            ),
            "mrr": reciprocal_rank(retrieved_ids, relevant),
        }

        results.append(result)

        print(f"\nQuestion: {result['question']}")
        print(f"Retrieved: {retrieved_ids}")
        print(f"Precision@K: {result['precision_at_k']:.3f}")
        print(f"Recall@K:    {result['recall_at_k']:.3f}")
        print(f"MRR:         {result['mrr']:.3f}")

    print("\n=== Mean Evaluation Metrics ===")

    for metric in ("precision_at_k", "recall_at_k", "mrr"):
        mean_value = sum(row[metric] for row in results) / len(results)
        print(f"{metric}: {mean_value:.3f}")


if __name__ == "__main__":
    main()
