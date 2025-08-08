import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class Config:
    """Application configuration"""
    
    # Flask settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    DEBUG: bool = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    HOST: str = os.getenv("FLASK_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("FLASK_PORT", "5000"))
    
    # API Keys
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    SERPER_API_KEY: str = os.getenv("SERPER_API_KEY", "")
    DEEPGRAM_API_KEY: str = os.getenv("DEEPGRAM_API_KEY", "")
    
    # LLM Model Configuration
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")
    TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "150"))
    
    # Panel Settings
    MAX_RESPONSE_WORDS: int = int(os.getenv("MAX_RESPONSE_WORDS", "200"))
    MAX_AGENT_RESPONSES: int = int(os.getenv("MAX_AGENT_RESPONSES", "3"))
    CONVERSATION_TIMEOUT: int = int(os.getenv("CONVERSATION_TIMEOUT", "1800"))
    
    # Memory Settings
    MEMORY_COLLECTION_NAME: str = os.getenv("MEMORY_COLLECTION_NAME", "panel_memory_web")
    MAX_MEMORY_ITEMS: int = int(os.getenv("MAX_MEMORY_ITEMS", "100"))
    MEMORY_THRESHOLD: float = float(os.getenv("MEMORY_THRESHOLD", "0.7"))
    
    # Session Settings
    SESSION_TIMEOUT: int = int(os.getenv("SESSION_TIMEOUT", "3600"))
    MAX_CONCURRENT_SESSIONS: int = int(os.getenv("MAX_CONCURRENT_SESSIONS", "50"))
    
    # CORS Settings - FIXED with default_factory
    CORS_ORIGINS: List[str] = field(default_factory=lambda: os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "panel_discussion.log")
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    RATE_LIMIT_PER_HOUR: int = int(os.getenv("RATE_LIMIT_PER_HOUR", "500"))

# Global config instance
config = Config()
