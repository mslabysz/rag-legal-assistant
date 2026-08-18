from pydantic import BaseModel, Field
from rag_legal_assistant.llm import llm
from rag_legal_assistant.prompts import GRADER_PROMPT, REWRITE_PROMPT

class Grade(BaseModel):
    binary_score: str = Field(description="Answer 'yes' or 'no' if the document answers the question")

grader_llm = llm.with_structured_output(Grade)
grader_chain = GRADER_PROMPT | grader_llm

rewrite_chain = REWRITE_PROMPT | llm
