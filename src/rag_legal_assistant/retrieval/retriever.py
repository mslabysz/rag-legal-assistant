import logging
from flashrank import Ranker
from rag_legal_assistant.config import settings
from rag_legal_assistant.llm import llm
from rag_legal_assistant.prompts import MULTI_QUERY_PROMPT
from rag_legal_assistant.vectordb.client import vector_store
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from qdrant_client.models import Filter, FieldCondition, MatchValue

logger = logging.getLogger(__name__)
query_llm = llm
ranker = Ranker(
    model_name=settings.RERANKER_MODEL,
    cache_dir=settings.RERANKER_CACHE_DIR,
)
compressor = FlashrankRerank(client=ranker, top_n=5)


def search(query: str, top_k: int, filter_document: str | None = None) -> list[dict]:
    search_kwargs = {"k": 20}

    if filter_document:
        search_kwargs["filter"] = Filter(
            must=[
                FieldCondition(
                    key="metadata.source",
                    match=MatchValue(value=filter_document)
                )
            ]
        )
    base_retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
    mq_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=query_llm,
        prompt=MULTI_QUERY_PROMPT,
        include_original=True,
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=mq_retriever
    )
    docs = compression_retriever.invoke(query)

    return [
        {
            "score": float(doc.metadata.get("relevance_score", 0)),
            "text": doc.page_content,
            "source": doc.metadata.get("source", "nieznane"),
            "chunk_index": doc.metadata.get("chunk_index", 0),
        }
        for doc in docs[:top_k]
    ]
