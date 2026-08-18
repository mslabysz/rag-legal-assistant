from typing import TypedDict, List

class GraphState(TypedDict):
    query: str
    documents: list[dict]
    answer: str
    retry_count: int