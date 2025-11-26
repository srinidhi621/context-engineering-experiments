"""Tests for model clients and monitoring"""
import pytest
from unittest.mock import MagicMock, patch
from src.models.gemini_client import GeminiClient
from src.utils.monitor import get_monitor, APIMonitor

@pytest.fixture
def mock_env_vars(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake_key")
    monkeypatch.setenv("RATE_LIMIT_RPM", "15")
    monkeypatch.setenv("RATE_LIMIT_TPM", "1000000")
    monkeypatch.setenv("RATE_LIMIT_RPD", "1500")

@patch("src.models.gemini_client.genai")
def test_client_initialization_monitors(mock_genai, mock_env_vars):
    """Test that GeminiClient initializes separate monitors for generation and embedding"""
    client = GeminiClient()
    
    assert client.monitor is not None
    assert client.embedding_monitor is not None
    assert client.monitor != client.embedding_monitor
    
    # Verify models
    assert client.monitor.model_name == "gemini-2.0-flash-exp"
    assert client.embedding_monitor.model_name == "text-embedding-004"
    
    # Verify state files are different (indirectly via internal state, or by checking private attrs if accessible)
    # In our implementation, get_monitor creates distinct instances with distinct state files
    assert client.monitor.state_file != client.embedding_monitor.state_file

@patch("src.models.gemini_client.genai")
def test_embed_text_uses_embedding_monitor(mock_genai, mock_env_vars):
    """Test that embed_text records calls to the embedding monitor"""
    client = GeminiClient()
    
    # Mock the embed_content response
    mock_genai.embed_content.return_value = {'embedding': [0.1, 0.2, 0.3]}
    
    # Mock the record_call method of the embedding monitor
    client.embedding_monitor.record_call = MagicMock()
    client.monitor.record_call = MagicMock()
    
    client.embed_text("test text")
    
    # Verify embedding monitor was called
    client.embedding_monitor.record_call.assert_called_once()
    
    # Verify generation monitor was NOT called
    client.monitor.record_call.assert_not_called()