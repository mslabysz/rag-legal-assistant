import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient
from rag_legal_assistant.api.routers import chat, documents
from rag_legal_assistant.config import settings
from rag_legal_assistant.ingestion.loader import load_all_documents
from rag_legal_assistant.chunking.chunker import chunk_document
from rag_legal_assistant.vectordb.client import index_documents

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    collections = [c.name for c in client.get_collections().collections]
    if settings.COLLECTION_NAME not in collections:
        logger.info("Collection not found. Starting automatic indexing...")
        docs = load_all_documents("data")
        all_chunks = []
        for d in docs:
            chunks = chunk_document(d["text"], d["source"], chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
            all_chunks.extend(chunks)
        index_documents(all_chunks, batch_size=50)
        logger.info("Indexing complete.")
    yield

app = FastAPI(
    title="Rag Legal Assistant API",
    description="API for the Agentic RAG system for Polish Law",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(chat.router)
app.include_router(documents.router)

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "rag-legal-assistant"}
