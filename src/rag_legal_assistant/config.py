from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    OPENAI_API_KEY: str
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    COLLECTION_NAME: str = "legal_docs"
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    EMBEDDING_MODEL: str = "sdadas/st-polish-paraphrase-from-mpnet"
    LLM_MODEL: str = "gpt-4o-mini"
    EMBEDDING_PROVIDER: str = "local"
    RERANKER_MODEL: str = "ms-marco-TinyBERT-L-2-v2"
    RERANKER_CACHE_DIR: str = "model_cache/flashrank"
    OPENAI_TIMEOUT: float = 60.0
    OPENAI_MAX_RETRIES: int = 2
    MAX_UPLOAD_MB: int = 25
    MAX_QUERY_REWRITES: int = 3

settings = Settings()