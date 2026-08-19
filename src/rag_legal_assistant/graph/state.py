from typing import TypedDict

class GraphState(TypedDict):
    query: str
    documents: list[dict]
    answer: str
    retry_count: int
    filter_document: str | None