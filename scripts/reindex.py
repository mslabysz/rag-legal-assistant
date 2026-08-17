import logging
logging.basicConfig(level=logging.INFO)
from qdrant_client import QdrantClient
from rag_legal_assistant.config import settings
from rag_legal_assistant.ingestion.loader import load_all_documents
from rag_legal_assistant.ingestion.chunker import chunk_document
from rag_legal_assistant.retrieval.retriever import index_documents

client = QdrantClient(host="localhost", port=6333)
client.delete_collection("legal_docs")
print("Old collection deleted")

docs = load_all_documents("data")
all_chunks = []
for d in docs:
    chunks = chunk_document(d["text"], d["source"], settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    all_chunks.extend(chunks)
print(f"chunks to index: {len(all_chunks)}")
print("This might take a while...")
index_documents(all_chunks, batch_size=50)
print("Done!")