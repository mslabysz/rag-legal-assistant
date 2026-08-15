import pymupdf
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def load_pdf(file_path: str) -> str:
    with pymupdf.open(file_path) as doc:
        pages = []
        for page in doc:
            text = page.get_text()
            pages.append(text)
    return "\n\n".join(pages)

def load_all_documents(directory: str) -> list[dict]:
    docs = []
    for pdf_path in Path(directory).glob("*.pdf"):
        text = load_pdf(str(pdf_path))
        docs.append({
            "source": pdf_path.name,
            "text": text,
        })
        logger.info(f"Loaded: {pdf_path.name} ({len(text)} chars)")
    return docs