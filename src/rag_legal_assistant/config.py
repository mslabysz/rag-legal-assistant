from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    OPENAI_API_KEY: str
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    COLLECTION_NAME: str = "legal_docs"
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    EMBEDDING_MODEL: str = "sdadas/st-polish-paraphrase-from-mpnet"
    LLM_MODEL: str = "gpt-4o-mini"
    EMBEDDING_PROVIDER: str = "local"

settings = Settings()