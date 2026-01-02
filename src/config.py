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
    # Primary model: Gemini 2.0 Flash-Lite (Cost-optimized, 1M context, 1500 RPD free tier)
    # June 2024 cutoff ensures no leakage of Sept-Dec 2024 corpus data
    model_name: str = "gemini-2.0-flash-lite-preview-02-05"
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
    # Free tier limits for gemini-2.0-flash-lite-preview-02-05
    rate_limit_rpm: int = int(os.getenv("RATE_LIMIT_RPM") or 30)
    rate_limit_tpm: int = int(os.getenv("RATE_LIMIT_TPM") or 4_000_000)
    rate_limit_rpd: int = int(os.getenv("RATE_LIMIT_RPD") or 1_500)
    # Budget limit (in USD) - hard stop to prevent overages
    budget_limit: float = float(os.getenv("BUDGET_LIMIT") or 174.00)
    # Safety switch: when set, block outgoing Gemini calls
    disable_calls: bool = os.getenv("GEMINI_DISABLE_CALLS", "").lower() in {"1", "true", "yes", "on"}

# Global config instances
config = ExperimentConfig()
api_config = APIConfig()
