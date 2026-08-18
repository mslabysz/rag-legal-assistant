import logging
from rag_legal_assistant.config import settings
from rag_legal_assistant.llm import llm
from rag_legal_assistant.vectordb.client import vector_store
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

logger = logging.getLogger(__name__)

base_retriever = vector_store.as_retriever(search_kwargs={"k": 20})

query_llm = llm

mq_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=query_llm
)

compressor = FlashrankRerank(top_n=5)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=mq_retriever
)


def search(query: str, top_k: int) -> list[dict]:
    docs = compression_retriever.invoke(query)

    return [
        {
            "score": doc.metadata.get("relevance_score", 0),
            "text": doc.page_content,
            "source": doc.metadata.get("source", "nieznane"),
            "chunk_index": doc.metadata.get("chunk_index", 0),
        }
        for doc in docs[:top_k]
    ]
