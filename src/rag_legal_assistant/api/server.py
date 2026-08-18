import logging
import json
import warnings
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, HTTPException
from rag_legal_assistant.api.schemas import ChatRequest, ChatResponse
from rag_legal_assistant.graph.builder import app as agent_app
from fastapi.middleware.cors import CORSMiddleware

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    from rag_legal_assistant.config import settings
    from qdrant_client import QdrantClient
    client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    collections = [c.name for c in client.get_collections().collections]
    if settings.COLLECTION_NAME not in collections:
        logger.info("Collection not found. Starting automatic indexing...")
        from rag_legal_assistant.retrieval.retriever import index_documents
        from rag_legal_assistant.ingestion.loader import load_all_documents
        from rag_legal_assistant.ingestion.chunker import chunk_document
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
                {"query": request.query, "retry_count": 0},
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