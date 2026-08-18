from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from rag_legal_assistant.llm import llm

class Grade(BaseModel):
    binary_score: str = Field(description="Answer 'yes' or 'no' if the document answers the question")

grader_llm = llm.with_structured_output(Grade)

grader_prompt = ChatPromptTemplate.from_template(
    """You are a strict grader evaluating the relevance of a retrieved document to a user question.
    If the document contains keywords or semantic meaning related to the question, grade it as "yes".
    Otherwise, grade it as "no".

    Question: {query}
    Document: {context}
    """
)

grader_chain = grader_prompt | grader_llm

rewrite_prompt = ChatPromptTemplate.from_template(
    """You are an expert question re-writer. Your task is to convert the user's question into a better version that is optimized for vector store retrieval in Polish law.
    Analyze the input and reason about the underlying semantic intent.

    Original question: {query}

    Rewritten question (in Polish):"""
)
rewrite_chain = rewrite_prompt | llm
