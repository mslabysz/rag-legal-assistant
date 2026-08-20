import logging
from pathlib import Path

import pymupdf
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from rag_legal_assistant.chunking.chunker import chunk_document
from rag_legal_assistant.config import settings
from rag_legal_assistant.ingestion.loader import load_single_document
from rag_legal_assistant.vectordb.client import index_documents

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Documents"])

DATA_DIR = Path("data")
COPY_CHUNK_BYTES = 1024 * 1024


def _safe_target_path(filename: str | None) -> Path:
    if not filename:
        raise HTTPException(status_code=400, detail="Brak nazwy pliku")
    name = Path(filename).name
    if name in {"", ".", ".."} or not name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Obsługiwane są wyłącznie pliki PDF")
    return DATA_DIR / name


def _store_upload(file: UploadFile, target: Path) -> None:
    limit = settings.MAX_UPLOAD_MB * 1024 * 1024
    written = 0
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as buffer:
        while data := file.file.read(COPY_CHUNK_BYTES):
            written += len(data)
            if written > limit:
                buffer.close()
                target.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"Plik przekracza limit {settings.MAX_UPLOAD_MB} MB",
                )
            buffer.write(data)
    if written == 0:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Przesłany plik jest pusty")


def _ingest(target: Path) -> int:
    doc = load_single_document(str(target), target.name)
    chunks = chunk_document(
        doc["text"], doc["source"], settings.CHUNK_SIZE, settings.CHUNK_OVERLAP
    )
    if not chunks:
        raise HTTPException(status_code=400, detail="Nie udało się wyciągnąć tekstu z PDF-a")
    index_documents(chunks, batch_size=50)
    return len(chunks)


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    target = _safe_target_path(file.filename)

    await run_in_threadpool(_store_upload, file, target)

    try:
        chunks_indexed = await run_in_threadpool(_ingest, target)
    except HTTPException:
        target.unlink(missing_ok=True)
        raise
    except (pymupdf.FileDataError, pymupdf.EmptyFileError) as exc:
        target.unlink(missing_ok=True)
        logger.warning(f"Rejected invalid PDF {target.name}: {exc}")
        raise HTTPException(status_code=400, detail="Plik nie jest poprawnym PDF-em") from exc
    except Exception as exc:
        target.unlink(missing_ok=True)
        logger.exception(f"Failed to index {target.name}")
        raise HTTPException(status_code=500, detail="Nie udało się zaindeksować dokumentu") from exc

    return {"status": "success", "filename": target.name, "chunks_indexed": chunks_indexed}


@router.get("/documents")
def get_documents():
    try:
        return {"documents": sorted(path.name for path in DATA_DIR.glob("*.pdf"))}
    except OSError as exc:
        logger.exception("Failed to read the data/ directory")
        raise HTTPException(
            status_code=500, detail="Nie udało się odczytać listy dokumentów"
        ) from exc