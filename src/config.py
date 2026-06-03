from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    cohere_api_key: str = Field(default="", env="COHERE_API_KEY")
    chroma_persist_dir: str = Field(default=".chroma", env="CHROMA_PERSIST_DIR")
    chunk_size: int = Field(default=512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=64, env="CHUNK_OVERLAP")
    top_k: int = Field(default=5, env="TOP_K")
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    collection_name: str = "documents"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
