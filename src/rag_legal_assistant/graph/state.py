from typing import TypedDict, List
from langchain_core.documents import Document

class GraphState(TypedDict):
    query: str
    documents: List[Document]
    answer: str
    retry_count: int