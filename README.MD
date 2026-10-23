# LangChain RAG Document Q&A

A Retrieval-Augmented Generation (RAG) application that answers questions from custom documents using semantic search and large language models (LLMs). Built with LangChain, FAISS, Hugging Face embeddings, Groq, and Gradio.

The application retrieves relevant document chunks and uses them as context to generate grounded answers, helping users explore information contained in their own documents.

## Features

- **Document ingestion:** Load and process text documents from a local data directory.
- **Semantic search:** Retrieve relevant document chunks using Hugging Face sentence-transformer embeddings.
- **FAISS vector store:** Store and search document embeddings efficiently.
- **Persistent indexing:** Save the vector store locally to avoid rebuilding it unnecessarily.
- **Configurable retrieval:** Adjust the number of retrieved chunks and relevance threshold.
- **Similarity and MMR retrieval:** Support similarity-based retrieval and Maximal Marginal Relevance (MMR), when configured.
- **Grounded answers:** Use retrieved document context to help the LLM answer questions.
- **Source previews:** Inspect retrieved text chunks and their relevance scores.
- **Source references:** Display information about the documents used to generate an answer.
- **Interactive interface:** Ask questions through a Gradio web interface.
- **Retrieval evaluation:** Evaluate retrieval quality using Precision@K, Recall@K, and Mean Reciprocal Rank (MRR).
- **Error handling and logging:** Improve debugging and handle invalid inputs and configuration issues.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Application development |
| LangChain | RAG pipeline and document retrieval |
| FAISS | Vector similarity search |
| Hugging Face Sentence Transformers | Text embeddings |
| Groq | LLM inference |
| Gradio | Interactive web interface |
| python-dotenv | Environment variable management |

## Project Structure

```text
langchain-rag-document-qa/
├── app.py
├── rag_pipeline.py
├── ingestion.py
├── evaluation.py
├── test_questions.json
├── requirements.txt
├── .env.example
├── .gitignore
├── data/
│   ├── attendance.txt
│   ├── grading.txt
│   └── course_schedule.txt
└── vector_store/
    └── .gitkeep
```

*The structure above describes the intended project layout. Adjust it if your actual files or directories differ.*

## How It Works

1. **Document loading:** Read text documents from the `data/` directory.
2. **Text splitting:** Divide documents into smaller chunks for retrieval.
3. **Embedding generation:** Convert text chunks into numerical vectors using a sentence-transformer model.
4. **Vector storage:** Save the vectors in a local FAISS index.
5. **Question processing:** Convert the user's question into an embedding and retrieve relevant chunks.
6. **Answer generation:** Send the question and retrieved context to the Groq-hosted LLM.
7. **Source inspection:** Display retrieved passages and their relevance scores in the interface.
8. **Evaluation:** Measure retrieval performance against a labeled question dataset.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/langchain-rag-document-qa.git
cd langchain-rag-document-qa
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root. You can use `.env.example` as a template.

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu

CHUNK_SIZE=500
CHUNK_OVERLAP=100

DEFAULT_TOP_K=3
MIN_RELEVANCE_SCORE=0.20
RETRIEVAL_MODE=similarity

LOG_LEVEL=INFO
```

Use a Groq model currently available to your account. Configuration names and default values should match those supported by your implementation.

**Security:** Never commit your actual `.env` file or API key to GitHub.

### 5. Add your documents

Place your supported text files in the `data/` directory. The example documents can be replaced with documents relevant to your use case.

### 6. Build the vector store

Run the ingestion script:

```bash
python ingestion.py
```

This should process your documents and create or update the local FAISS index, according to your implementation.

### 7. Launch the application

```bash
python app.py
```

Open the local URL printed in your terminal by Gradio.

### 8. Evaluate retrieval

After configuring your evaluation questions and confirming that their expected chunk IDs match your generated index, run:

```bash
python evaluation.py
```

Review Precision@K, Recall@K, and MRR to understand how well the retriever finds relevant chunks.

## Evaluation

The project includes an evaluation workflow for:

- **Precision@K:** The proportion of retrieved chunks that are relevant.
- **Recall@K:** The proportion of known relevant chunks that are retrieved.
- **Mean Reciprocal Rank (MRR):** How highly the first relevant result is ranked.

Actual scores depend on the evaluation dataset, chunking strategy, embedding model, and retrieval configuration. Add measured results here after running your evaluation; do not report unmeasured performance claims.

## Limitations

- The application initially supports the document formats implemented in the ingestion pipeline.
- Answer quality depends on the source documents, retrieval quality, and selected LLM.
- Relevance scores are retrieval signals, not guarantees that an answer is correct.
- The initial evaluation dataset may require additional labeled questions to provide meaningful results.

## Future Improvements

- Add PDF and DOCX document support.
- Add automated tests for ingestion and retrieval.
- Improve citation formatting with document names and chunk references.
- Experiment with reranking and hybrid search.
- Expand the evaluation dataset and compare retrieval configurations.
- Deploy the application for public access.

## Learning Objectives

This project demonstrates practical experience with:

- Retrieval-Augmented Generation (RAG)
- Embeddings and vector databases
- Semantic search and document chunking
- LLM integration
- Retrieval evaluation
- Environment configuration and API-key security
- Building an interactive AI application

## Author

**Your Name**

- GitHub: [YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- LinkedIn: [Add your LinkedIn profile URL]

## License

This project is available under the MIT License if you choose to include an MIT `LICENSE` file.
