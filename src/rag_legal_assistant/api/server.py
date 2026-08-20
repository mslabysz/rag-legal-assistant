import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from rag_legal_assistant.logging_config import setup_logging
from rag_legal_assistant.api.routers import chat, documents
from rag_legal_assistant.chunking.chunker import chunk_document
from rag_legal_assistant.config import settings
from rag_legal_assistant.ingestion.loader import load_all_documents
from rag_legal_assistant.vectordb.client import client, index_documents

setup_logging()
logger = logging.getLogger(__name__)

ALLOWED_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


async def _wait_for_qdrant(attempts: int = 15, delay: float = 2.0) -> list[str]:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return [c.name for c in client.get_collections().collections]
        except Exception as exc:
            last_error = exc
            logger.warning(f"Qdrant not ready yet ({attempt}/{attempts}): {exc}")
            await asyncio.sleep(delay)
    raise RuntimeError("Qdrant is unreachable, aborting API startup") from last_error


def _index_corpus() -> None:
    all_chunks = []
    for doc in load_all_documents("data"):
        all_chunks.extend(
            chunk_document(
                doc["text"], doc["source"], settings.CHUNK_SIZE, settings.CHUNK_OVERLAP
            )
        )
    index_documents(all_chunks, batch_size=50)


@asynccontextmanager
async def lifespan(app: FastAPI):
    collections = await _wait_for_qdrant()
    if settings.COLLECTION_NAME not in collections:
        logger.info("Collection not found, indexing the corpus...")
        await run_in_threadpool(_index_corpus)
        logger.info("Indexing complete.")
    yield


app = FastAPI(
    title="Rag Legal Assistant API",
    description="API for the Agentic RAG system for Polish Law",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(documents.router)


@app.get("/health")
async def health_check():
    try:
        await run_in_threadpool(client.get_collections)
    except Exception as exc:
        logger.warning(f"Health check failed: {exc}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "service": "rag-legal-assistant",
                "qdrant": "unreachable",
            },
        )
    return {"status": "ok", "service": "rag-legal-assistant", "qdrant": "ok"}