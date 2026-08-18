import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

def chunk_document(text: str, source: str, chunk_size: int, chunk_overlap: int) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ",""]
    )

    chunks = [
        {"text": chunk_text, "source": source, "chunk_index": i}
        for i, chunk_text in enumerate(splitter.split_text(text))
    ]

    logger.info(f"{source}: {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks