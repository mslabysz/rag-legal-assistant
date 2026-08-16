import logging
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, ScoredPoint
from rag_legal_assistant.config import settings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_qdrant import QdrantVectorStore
from langchain_openai import ChatOpenAI
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

logger = logging.getLogger(__name__)

client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

if settings.EMBEDDING_PROVIDER == "local":
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )
    VECTOR_SIZE = 768
else:
    embeddings = OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY,
    )
    VECTOR_SIZE = 1536

def _ensure_collection_exists():
    collections = [c.name for c in client.get_collections().collections]

    if settings.COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=settings.COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )
        logger.info(f"Created collection {settings.COLLECTION_NAME}")

def index_documents(chunks: list[dict], batch_size: int = 100):
    _ensure_collection_exists()

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
                    f" points ({batch_start + len(batch)}/{len(chunks)}")
    logger.info(f"Indexed {len(chunks)} chunks")


def search(query: str, top_k: int) -> list[dict]:
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=settings.COLLECTION_NAME,
        embedding=embeddings,
        content_payload_key="text",
    )

    base_retriever = vector_store.as_retriever(search_kwargs={"k": 20})

    query_llm = ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.OPENAI_API_KEY
    )

    mq_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=query_llm
    )

    compressor = FlashrankRerank(top_n=top_k)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=mq_retriever
    )

    docs = compression_retriever.invoke(query)

    return [
        {
            "score": doc.metadata.get("relevance_score", 0),
            "text": doc.page_content,
            "source": doc.metadata.get("source", "nieznane"),
            "chunk_index": doc.metadata.get("chunk_index", 0),
        }
        for doc in docs
    ]

