import logging
logging.basicConfig(level=logging.INFO)
from rag_legal_assistant.config import settings
from rag_legal_assistant.ingestion.loader import load_all_documents
from rag_legal_assistant.ingestion.chunker import chunk_document
from rag_legal_assistant.retrieval.retriever import index_documents, search
from qdrant_client import QdrantClient

docs = load_all_documents("data")
all_chunks = []
for d in docs:
    chunks = chunk_document(d["text"], d["source"], chunk_size=1500, chunk_overlap=200)
    all_chunks.extend(chunks)
print(f"chunks: {len(all_chunks)}")

client = QdrantClient(host="localhost", port=6333)
client.delete_collection("legal_docs")
print("old collection deleted")
index_documents(all_chunks)

query = "Jaki jest termin przedawnienia roszczeń?"
results = search(query, top_k=3)
print(f"\noutput for: '{query}'")
for r in results:
    print(f"\nScore: {r['score']:.4f} | Source: {r['source']}")
    print(f"Text: {r['text'][:200]}...")