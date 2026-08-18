from langchain_core.prompts import ChatPromptTemplate

GRADER_PROMPT = ChatPromptTemplate.from_template(
    """You are a strict grader evaluating the relevance of a retrieved document to a user question.
    If the document contains keywords or semantic meaning related to the question, grade it as "yes".
    Otherwise, grade it as "no".

    Question: {query}
    Document: {context}
    """
)

REWRITE_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert question re-writer. Your task is to convert the user's question into a better version that is optimized for vector store retrieval in Polish law.
    Analyze the input and reason about the underlying semantic intent.

    Original question: {query}

    Rewritten question (in Polish):"""
)

GENERATOR_PROMPT = ChatPromptTemplate.from_template(
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
