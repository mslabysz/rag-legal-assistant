import asyncio
import logging
from rag_legal_assistant.graph.state import GraphState
from rag_legal_assistant.retrieval.retriever import search
from rag_legal_assistant.graph.chains import grader_chain, rewrite_chain
from rag_legal_assistant.llm import llm_streaming
from rag_legal_assistant.prompts import GENERATOR_PROMPT, NO_CONTEXT_ANSWER

logger = logging.getLogger(__name__)

chain = GENERATOR_PROMPT | llm_streaming

def retrieve_node(state: GraphState) :
    logger.info("---NODE: RETRIEVE ---")

    question = state["query"]
    docs = search(question, top_k=5, filter_document=state.get("filter_document"))
    return {"documents": docs}

async def generate_answer_node(state: GraphState):
    logger.info("---NODE: GENERATE ANSWER ---")

    question = state["query"]
    docs = state["documents"]

    if not docs:
        logger.warning("No documents survived grading, refusing to answer without context")
        return {"answer": NO_CONTEXT_ANSWER}

    context = "\n\n---\n\n".join(
        f"[Source: {doc.get('source', 'nieznane')}]\n{doc.get('text', '')}"
        for doc in docs
    )

    response = await chain.ainvoke({"query": question, "context": context})
    return {"answer": response.content}

async def grade_document_node(state: GraphState):
    logger.info("---NODE: GRADING ---")

    tasks = [
        grader_chain.ainvoke({"query": state["query"], "context": doc.get("text")})
        for doc in state["documents"]
    ]
    grades = await asyncio.gather(*tasks)
    filtered = [doc for doc, grade in zip(state["documents"], grades) if grade.binary_score.strip().lower() == "yes"]
    return {"documents": filtered}

def rewrite_query_node(state: GraphState):
    logger.info("---NODE: REWRITING ---")

    question = state["query"]
    response = rewrite_chain.invoke({"query": question})
    current_retries = state.get("retry_count", 0)
    return {"query": response.content, "retry_count": current_retries + 1}