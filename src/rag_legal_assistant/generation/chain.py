import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from rag_legal_assistant.config import settings
from rag_legal_assistant.retrieval.retriever import search

logger = logging.getLogger(__name__)

llm = ChatOpenAI(
    model=settings.LLM_MODEL,
    api_key=settings.OPENAI_API_KEY,
    temperature=0
)

prompt = ChatPromptTemplate.from_template(
    """You are a legal assistant specializing in Polish law.
Answer the user's question based ONLY on the provided context.
If the context does not contain enough information to answer, say so clearly.
Always cite the source document when possible.
Respond in Polish.

Context:
{context}

Question: {query}

Answer:"""
)

chain = prompt | llm

def generate_answer(query: str, top_k: int = 5) -> str:
    results = search(query, top_k=top_k)
    logger.info(f"Retrieved {len(results)} chunks for query: '{query}'")

    context = "\n\n---\n\n".join(
        f"[Source: {r["source"]}]\n{r['text']}"
        for r in results
    )

    response = chain.invoke({"context": context, "query": query})
    logger.info(f"Generated answer ({len(response.content)} chars)")
    return response.content
