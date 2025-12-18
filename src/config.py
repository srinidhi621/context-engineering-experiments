"""Configuration for experiments"""
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class ExperimentConfig:
    """Base configuration for all experiments"""
    # Primary model: Gemini Flash Latest (latest stable Flash model with 8x more output)
    # 15 RPM, 1M TPM, 1500 RPD - free tier with 65K output tokens vs 8K in 2.0-flash-exp
    model_name: str = "models/gemini-2.0-flash"
    # Embedding model: Latest production text embedding model
    embedding_model_name: str = "text-embedding-004"
    context_limit: int = 1_000_000
    temperature: float = 0.0
    repetitions: int = 3
    fill_percentages: list = None
    
    def __post_init__(self):
        if self.fill_percentages is None:
            self.fill_percentages = [0.1, 0.3, 0.5, 0.7, 0.9]

@dataclass
class APIConfig:
    """API configuration"""
    google_api_key: str = os.getenv("GOOGLE_API_KEY") or ""
    # Free tier limits for gemini-2.0-flash-exp
    rate_limit_rpm: int = int(os.getenv("RATE_LIMIT_RPM") or 15)
    rate_limit_tpm: int = int(os.getenv("RATE_LIMIT_TPM") or 1_000_000)
    rate_limit_rpd: int = int(os.getenv("RATE_LIMIT_RPD") or 1_000)
    # Budget limit (in USD) - hard stop to prevent overages
    budget_limit: float = float(os.getenv("BUDGET_LIMIT") or 174.00)

# Global config instances
config = ExperimentConfig()
api_config = APIConfig()

