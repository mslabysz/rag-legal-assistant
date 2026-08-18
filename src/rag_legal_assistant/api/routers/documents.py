import os
import shutil
import logging
from fastapi import APIRouter, HTTPException, File, UploadFile
from rag_legal_assistant.config import settings
from rag_legal_assistant.vectordb.client import index_documents, client
from rag_legal_assistant.ingestion.loader import load_single_document
from rag_legal_assistant.chunking.chunker import chunk_document

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Documents"])

@router.post("/upload")
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
@router.get("/documents")
def get_documents():
    docs = []
    try:
        if os.path.exists("data"):
            for f in os.listdir("data"):
                if f.endswith(".pdf"):
                    docs.append(f)
        return {"documents": docs}
    except Exception as e:
        logger.error(f"Error fetching documents: {e}")
        return {"documents": []}