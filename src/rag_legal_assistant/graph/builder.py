from langgraph.graph import StateGraph, END
from rag_legal_assistant.graph.state import GraphState
from rag_legal_assistant.graph.nodes import retrieve_node, generate_answer_node, grade_document_node, rewrite_query_node

workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", grade_document_node)
workflow.add_node("generate_answer", generate_answer_node)
workflow.add_node("rewrite_query", rewrite_query_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_edge("rewrite_query", "retrieve")
workflow.add_edge("generate_answer", END)

def decide_to_generate(state: GraphState):
    docs = state["documents"]
    retries = state.get("retry_count", 0)
    if len(docs)>0 or retries>=3:
        return "generate_answer"
    else:
        return "rewrite_query"

workflow.add_conditional_edges("grade_documents", decide_to_generate)

app = workflow.compile()