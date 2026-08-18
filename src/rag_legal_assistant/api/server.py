import logging
import json
import warnings
import os
import shutil
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, HTTPException, File, UploadFile
from rag_legal_assistant.api.schemas import ChatRequest, ChatResponse
from rag_legal_assistant.graph.builder import app as agent_app
from fastapi.middleware.cors import CORSMiddleware
from rag_legal_assistant.config import settings
from qdrant_client import QdrantClient
from rag_legal_assistant.vectordb.client import index_documents
from rag_legal_assistant.ingestion.loader import load_all_documents, load_single_document
from rag_legal_assistant.chunking.chunker import chunk_document

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager

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

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "rag-legal-assistant"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    logger.info(f"Received query: {request.query}")

    try:
        final_state = await agent_app.ainvoke(
            {"query": request.query, "retry_count": 0}
        )
        return ChatResponse(
            answer=final_state["answer"],
            retries=final_state.get("retry_count", 0)
        )

    except Exception as e:
        logger.error(f"Error during agent execution: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    async def event_generator():
        try:
            is_generating = False
            retry_count = 0
            async for event in agent_app.astream_events(
                {"query": request.query,
                 "retry_count": 0,
                 "filter_document": request.filter_document
                 },
                version="v2"
            ):
                kind = event["event"]
                name = event["name"]
                
                if kind == "on_chain_start":
                    if name == "retrieve":
                        yield f"data: {json.dumps({'type': 'status', 'message': 'Szukam dokumentów w bazie...'})}\n\n"
                    elif name == "grade_documents":
                        yield f"data: {json.dumps({'type': 'status', 'message': 'Sędzia ocenia przydatność dokumentów...'})}\n\n"
                    elif name == "rewrite_query":
                        retry_count += 1
                        yield f"data: {json.dumps({'type': 'status', 'message': 'Odrzucono dokumenty. Przepisuję zapytanie i ponawiam próbę...'})}\n\n"
                    elif name == "generate_answer":
                        yield f"data: {json.dumps({'type': 'status', 'message': 'Piszę odpowiedź...'})}\n\n"
                        is_generating = True
                        
                        state_input = event.get("data", {}).get("input", {})
                        if isinstance(state_input, dict) and "documents" in state_input:
                            yield f"data: {json.dumps({'type': 'sources', 'documents': state_input['documents']})}\n\n"
                        
                elif kind == "on_chain_end":
                    if name == "generate_answer":
                        is_generating = False
                        
                elif kind == "on_chat_model_stream" and is_generating:
                    chunk = event["data"]["chunk"]
                    if chunk.content:
                        yield f"data: {json.dumps({'type': 'chunk', 'text': chunk.content})}\n\n"
            
            yield f"data: {json.dumps({'type': 'done', 'retries': retry_count})}\n\n"
            
        except Exception as e:
            logger.error(f"Error during agent stream: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        file_path = os.path.join("data", file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        doc = load_single_document(file_path, file.filename)
        chunks = chunk_document(doc["text"], doc["source"], settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)

        index_documents(chunks, batch_size=50)

        return {"status": "success", "chunks_indexed": len(chunks)}

    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error during upload")


@app.get("/documents")
def get_documents():
    from rag_legal_assistant.config import settings
    from rag_legal_assistant.vectordb.client import client

    try:
        res = client.scroll(
            collection_name=settings.COLLECTION_NAME,
            limit=1000,
            with_payload=True,
            with_vectors=False
        )
        points = res[0]
        unique_docs = list(set([p.payload.get("metadata", {}).get("source") for p in points if
                                p.payload and p.payload.get("metadata")]))
        return {"documents": [d for d in unique_docs if d]}
    except Exception as e:
        return {"documents": []}