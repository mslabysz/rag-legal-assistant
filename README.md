# RAG Legal Assistant

An advanced, agentic Retrieval-Augmented Generation (RAG) system specialized in the analysis of Polish law. Built with LangGraph, FastAPI, and React, it features a self-corrective retrieval pipeline, real-time Server-Sent Events (SSE) streaming, cross-encoder reranking, and dynamic document upload capabilities.

## Features

### Self-Corrective Retrieval Pipeline
- Implemented using a state machine via **LangGraph**.
- The pipeline does not blindly trust the retrieved documents. Instead, it utilizes an LLM-as-a-Judge (Grader) to evaluate the relevance of the retrieved context against the user query.
- If the retrieved context is deemed irrelevant, a Rewriter node autonomously reformulates the query and triggers a new retrieval cycle until a satisfactory context is found or a retry limit is reached.

### Advanced RAG Architecture
- **Semantic Search**: Utilizes Qdrant vector database for high-performance similarity search.
- **Cross-Encoder Reranking**: Employs **FlashRank** to drastically improve retrieval precision by re-scoring and re-ordering chunks based on deep semantic overlap.
- **Multi-Query Retrieval**: Automatically generates multiple semantic variants of the user query to maximize context recall and overcome vocabulary mismatch in legal documents.
- **Metadata Filtering**: Enables users to narrow down the vector search space to specific, user-selected legal documents, preventing cross-document hallucination.

### Dynamic Knowledge Base & Citations
- **In-browser PDF Upload**: Users can seamlessly upload new PDF documents via the chat interface. The backend automatically parses, chunks, embeds, and indexes the document into Qdrant on-the-fly.
- **Source Citations Panel**: The frontend visually displays the exact source documents, chunk text, and Reranker relevance scores used by the AI to generate the answer, providing full explainability (Explainable AI).

### Real-Time Streaming
- Fully asynchronous backend built with FastAPI and cleanly organized using `APIRouter`.
- Employs Server-Sent Events (SSE) via LangGraph's `astream_events` (v2) to stream both the retrieved sources and the generated answer token-by-token directly to the React frontend, ensuring a highly responsive user experience.

### Quantitative Evaluation
- Pipeline accuracy and robustness rigorously evaluated using the **RAGAS** framework.
- Evaluated against a custom ground-truth dataset comprising complex legal scenarios.

## Performance & Evaluation

The system was benchmarked using the **RAGAS** (Retrieval Augmented Generation Assessment) framework utilizing an LLM-as-a-Judge methodology on a dedicated legal dataset.

- **Faithfulness**: `0.9375` (Answers are highly grounded in the retrieved context with minimal hallucination)
- **Answer Relevancy**: `0.8502` (Responses directly address the user's intent)
- **Context Precision**: `0.7500` (Highly relevant documents are ranked at the top of the retrieval results)
- **Context Recall**: `0.7500` (The retrieved context contains the necessary information to answer the query)

## Design Decisions

1. **Why LangGraph over standard LangChain Chains?**
   Standard LCEL chains are linear (DAGs) and do not support cyclic execution. By utilizing LangGraph, the system implements a cyclic, self-correcting loop where poor retrieval results trigger a query rewrite and a subsequent re-retrieval. This significantly improves accuracy on complex legal queries where initial keyword matching often fails.
2. **Why Qdrant?**
   Qdrant was chosen over FAISS for its robust Docker support, production-readiness, and built-in REST/gRPC APIs, allowing for seamless integration into a containerized microservices architecture.
3. **Why uv?**
   `uv` by Astral was selected as the package manager due to its Rust-based dependency resolution, which drastically reduces build times during Docker image construction compared to standard `pip`.

## Tech Stack

| Layer | Technologies |
|---|---|
| **Backend** | FastAPI, APIRouter, Python 3.12, uv |
| **Vector Database** | Qdrant |
| **Orchestration** | LangGraph, LangChain |
| **Reranking** | FlashRank |
| **LLM & Embeddings** | OpenAI (gpt-4o-mini), HuggingFace (`st-polish-paraphrase-from-mpnet`) |
| **Frontend** | React 19, Vite, TypeScript, Tailwind CSS |
| **Deployment** | Docker, Docker Compose (with local HF cache volume) |

## Architecture

```mermaid
graph TD
    User([User]) -->|Query + Optional Filter| Frontend[React SPA]
    User -->|Upload PDF| Frontend
    
    Frontend -->|POST /upload| RouterDocs[Documents Router]
    RouterDocs -->|Chunk & Embed| Qdrant[(Qdrant Vector Store)]
    
    Frontend -->|POST /chat/stream| RouterChat[Chat Router]
    
    subgraph LangGraph Pipeline
        RouterChat --> Retrieve[Retrieve Node]
        Retrieve -->|Multi-Query| Qdrant
        Qdrant -->|Initial Chunks| Reranker[FlashRank Reranker]
        Reranker -->|Top-K Chunks| Grade[Grade Relevance Node]
        
        Grade -->|Irrelevant?| Rewrite[Rewrite Query Node]
        Rewrite -->|New Query| Retrieve
        
        Grade -->|Relevant?| Generate[Generate Answer Node]
    end
    
    Generate -->|SSE Token Stream + Sources| RouterChat
    RouterChat -->|SSE Token Stream| Frontend
```

## Getting Started

### Prerequisites
- Docker and Docker Compose
- An OpenAI API key

### Installation and Execution

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/rag-legal-assistant.git
   cd rag-legal-assistant
   ```

2. Create a `.env` file in the root directory and add your OpenAI API key:
   ```env
   OPENAI_API_KEY=sk-your-openai-api-key
   ```

3. Build and run the containers using Docker Compose:
   ```bash
   docker compose up --build -d
   ```
   *Note: On the first run, the system will download the embedding and reranker models to a local `./hf_cache` volume to persist them across restarts.*

The orchestration spins up three services:
- **Qdrant**: Available at `http://localhost:6333`
- **FastAPI Backend**: Available at `http://localhost:8000` (Interactive API docs at `http://localhost:8000/docs`)
- **React Frontend**: Available at `http://localhost:5173`

Navigate to `http://localhost:5173` in your browser to interact with the Legal Assistant.

## License

MIT
