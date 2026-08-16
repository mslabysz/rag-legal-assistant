import logging
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, ScoredPoint
from rag_legal_assistant.config import settings

logger = logging.getLogger(__name__)

client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

embeddings = OpenAIEmbeddings(
    model=settings.EMBEDDING_MODEL,
    api_key=settings.OPENAI_API_KEY
)

def _ensure_collection_exists():
    collections = [c.name for c in client.get_collections().collections]

    if settings.COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=settings.COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
        logger.info(f"Created collection {settings.COLLECTION_NAME}")

def index_documents(chunks: list[dict], batch_size: int = 100):
    _ensure_collection_exists()

    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start:batch_start + batch_size]

        texts = [chunk["text"] for chunk in chunks]
        vectors = embeddings.embed_documents(texts)

        points = [
            PointStruct(
                id=batch_start+i,
                vector=vector,
                payload={
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "chunk_index": chunk["chunk_index"],
                }
            )
            for i, (chunk,vector) in enumerate(zip(batch,vectors))
        ]

        client.upsert(
            collection_name=settings.COLLECTION_NAME,
            points=points
        )
        logger.info(f"Batch {batch_start // batch_size+1}: saved {len(points)}"
                    f" points ({batch_start + len(batch)}/{len(chunks)}")
    logger.info(f"Indexed {len(chunks)} chunks")


def search(query: str, top_k: int) -> list[dict]:
    query_vector = embeddings.embed_query(query)

    results = client.query_points(
        collection_name=settings.COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points

    return [
        {
            "score": hit.score,
            "text": hit.payload["text"],
            "source": hit.payload["source"],
            "chunk_index": hit.payload["chunk_index"],
        }
        for hit in results
    ]