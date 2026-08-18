from langchain_openai import ChatOpenAI
from rag_legal_assistant.config import settings

llm = ChatOpenAI(
    model=settings.LLM_MODEL,
    api_key=settings.OPENAI_API_KEY,
    temperature=0
)

llm_streaming = ChatOpenAI(
    model=settings.LLM_MODEL,
    api_key=settings.OPENAI_API_KEY,
    temperature=0,
    streaming=True
)
