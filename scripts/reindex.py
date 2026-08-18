import logging
logging.basicConfig(level=logging.INFO)
from qdrant_client import QdrantClient
from rag_legal_assistant.config import settings
from rag_legal_assistant.ingestion.loader import load_all_documents
from rag_legal_assistant.chunking.chunker import chunk_document
from rag_legal_assistant.retrieval.retriever import index_documents

client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

collections = [c.name for c in client.get_collections().collections]
if settings.COLLECTION_NAME in collections:
    client.delete_collection(settings.COLLECTION_NAME)
    print(f"Old collection '{settings.COLLECTION_NAME}' deleted")
else:
    print(f"Collection '{settings.COLLECTION_NAME}' does not exist, creating new...")

docs = load_all_documents("data")
all_chunks = []
for d in docs:
    chunks = chunk_document(d["text"], d["source"], settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    all_chunks.extend(chunks)
print(f"chunks to index: {len(all_chunks)}")
print("This might take a while...")
index_documents(all_chunks, batch_size=50)
print("Done!")