import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_qdrant import QdrantVectorStore
from rag_legal_assistant.config import settings
from rag_legal_assistant.embeddings.embedder import embeddings, VECTOR_SIZE

logger = logging.getLogger(__name__)

client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

vector_store = QdrantVectorStore(
    client=client,
    collection_name=settings.COLLECTION_NAME,
    embedding=embeddings,
    content_payload_key="text",
)

def ensure_collection_exists():
    collections = [c.name for c in client.get_collections().collections]

    if settings.COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=settings.COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )
        logger.info(f"Created collection {settings.COLLECTION_NAME}")

def index_documents(chunks: list[dict], batch_size: int = 100):
    ensure_collection_exists()

    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start:batch_start + batch_size]

        texts = [chunk["text"] for chunk in batch]
        vectors = embeddings.embed_documents(texts)

        points = [
            PointStruct(
                id=batch_start+i,
                vector=vector,
                payload={
                    "text": chunk["text"],
                    "metadata": {
                        "source": chunk["source"],
                        "chunk_index": chunk["chunk_index"],
                    }
                }
            )
            for i, (chunk,vector) in enumerate(zip(batch,vectors))
        ]

        client.upsert(
            collection_name=settings.COLLECTION_NAME,
            points=points
        )
        logger.info(f"Batch {batch_start // batch_size+1}: saved {len(points)}"
                    f" points ({batch_start + len(batch)}/{len(chunks)})")
    logger.info(f"Indexed {len(chunks)} chunks")
