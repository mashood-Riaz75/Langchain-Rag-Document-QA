
import hashlib
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_documents():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    documents = []

    for path in sorted(DATA_DIR.glob("*.txt")):
        content = path.read_text(encoding="utf-8").strip()

        if not content:
            logger.warning("Skipping empty document: %s", path.name)
            continue

        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": path.name,
                    "document_sha256": digest,
                },
            )
        )

    if not documents:
        raise ValueError(f"No non-empty .txt documents found in {DATA_DIR}")

    logger.info("Loaded %d documents", len(documents))
    return documents


def build_vector_store():
    if CHUNK_SIZE <= 0:
        raise ValueError("CHUNK_SIZE must be positive.")

    if not 0 <= CHUNK_OVERLAP < CHUNK_SIZE:
        raise ValueError(
            "CHUNK_OVERLAP must be at least 0 and smaller than CHUNK_SIZE."
        )

    documents = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)
    per_source_count = {}

    for chunk in chunks:
        source = chunk.metadata["source"]
        number = per_source_count.get(source, 0)

        chunk.metadata["chunk_id"] = f"{source}::chunk_{number}"
        per_source_count[source] = number + 1

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": os.getenv("EMBEDDING_DEVICE", "cpu")},
        encode_kwargs={"normalize_embeddings": True},
    )

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    vector_store.save_local(str(VECTOR_STORE_DIR))

    manifest = {
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "chunk_count": len(chunks),
        "documents": [
            {
                "source": document.metadata["source"],
                "sha256": document.metadata["document_sha256"],
            }
            for document in documents
        ],
    }

    (VECTOR_STORE_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    logger.info(
        "Index rebuilt successfully: %d chunks from %d documents",
        len(chunks),
        len(documents),
    )


if __name__ == "__main__":
    try:
        build_vector_store()
    except Exception:
        logger.exception("Document ingestion failed")
        raise
