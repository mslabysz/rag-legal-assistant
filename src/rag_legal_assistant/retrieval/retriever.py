from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

def index_documents(chunks: list[dict]):
    pass

def search(query: str, top_k: int) -> list[dict]:
    pass