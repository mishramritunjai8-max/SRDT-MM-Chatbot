import os
from typing import List, Union
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator


class Settings(BaseSettings):
    APP_NAME: str = "Multimodal AI Chatbot MVP"
    APP_ENV: str = "development"
    DEBUG: bool = True
    PORT: int = 8000
    HOST: str = "127.0.0.1"

    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:8501", "http://127.0.0.1:8501", "*"]

    # AI Model Selection & Keys
    DEFAULT_MODEL: str = "auto"

    # Google Gemini
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3.5-flash"
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-001"

    # xAI Grok (https://console.x.ai/ - $25 free monthly credits)
    GROK_API_KEY: str = ""
    GROK_BASE_URL: str = "https://api.x.ai/v1"
    GROK_MODEL: str = "grok-2"
    GROK_VISION_MODEL: str = "grok-2-vision-1212"

    # Groq (100% Free Forever at https://console.groq.com/)
    GROQ_API_KEY: str = ""
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # Database
    DATABASE_URL: str = "sqlite:///./mm_chatbot.db"

    # RAG Settings
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K_RETRIEVAL: int = 4
    UPLOAD_DIR: str = "./uploads"

    # Web Search
    SEARCH_PROVIDER: str = "duckduckgo"
    MAX_SEARCH_RESULTS: int = 5

    # Frontend connection
    BACKEND_API_URL: str = "http://127.0.0.1:8000"

    @field_validator("CORS_ORIGINS", mode="before")
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, list):
            return v
        return ["*"]

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }


settings = Settings()

# Ensure uploads directory exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
