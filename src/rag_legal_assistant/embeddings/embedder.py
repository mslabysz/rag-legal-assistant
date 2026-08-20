import logging
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from rag_legal_assistant.config import settings

logger = logging.getLogger(__name__)

if settings.EMBEDDING_PROVIDER == "local":
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )
    VECTOR_SIZE = 768
else:
    embeddings = OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY,
    )
    VECTOR_SIZE = 1536
