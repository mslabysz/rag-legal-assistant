import logging
from rag_legal_assistant.graph.state import GraphState
from rag_legal_assistant.retrieval.retriever import search
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from rag_legal_assistant.config import settings
from rag_legal_assistant.graph.schemas import grader_chain
from rag_legal_assistant.graph.schemas import rewrite_chain

logger = logging.getLogger(__name__)

llm = ChatOpenAI(
    model=settings.LLM_MODEL,
    api_key=settings.OPENAI_API_KEY,
    temperature=0,
    streaming=True
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

def retrieve_node(state: GraphState) :
    logger.info("---NODE: RETRIEVE ---")

    question = state["query"]
    docs = search(question, top_k=5)
    return {"documents": docs}

async def generate_answer_node(state: GraphState):
    logger.info("---NODE: GENERATE ANSWER ---")

    question = state["query"]
    docs = state["documents"]
    context = "\n\n---\n\n".join(
        f"[Source: {doc.get('source', 'nieznane')}]\n{doc.get('text', '')}"
        for doc in docs
    )

    response = await chain.ainvoke({"query": question, "context": context})
    return {"answer": response.content}

def grade_document_node(state: GraphState):
    logger.info("---NODE: GRADING ---")

    question = state["query"]
    documents = state["documents"]
    filtered_docs = []

    for doc in documents:
        grade = grader_chain.invoke({"query": question, "context": doc.get("text")})
        if grade.binary_score == "yes":
            filtered_docs.append(doc)

    return {"documents": filtered_docs}

def rewrite_query_node(state: GraphState):
    logger.info("---NODE: REWRITING ---")

    question = state["query"]
    response = rewrite_chain.invoke({"query": question})
    current_retries = state.get("retry_count", 0)
    return {"query": response.content, "retry_count": current_retries + 1}