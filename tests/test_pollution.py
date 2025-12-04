"""Tests for PollutionInjector"""
import pytest
from src.corpus.pollution import PollutionInjector
from src.utils.tokenizer import count_tokens

class TestPollutionInjector:
    
    @pytest.fixture
    def injector(self):
        return PollutionInjector()
    
    def test_inject_pollution_append(self, injector):
        base = "This is the relevant base content."
        pollution = "This is irrelevant pollution content that goes on and on."
        target_tokens = 5
        
        result = injector.inject_pollution(base, pollution, target_tokens, strategy='append')
        
        assert base in result
        # Should contain some pollution
        assert len(result) > len(base)
        
        # Check token count roughly
        base_tokens = count_tokens(base)
        total_tokens = count_tokens(result)
        
        # We added ~5 tokens
        assert total_tokens >= base_tokens + 4 
        assert total_tokens <= base_tokens + 6

    def test_zero_pollution(self, injector):
        base = "Content"
        pollution = "Pollution"
        result = injector.inject_pollution(base, pollution, 0)
        assert result == base

    def test_strategy_prepend(self, injector):
        base = "Base"
        pollution = "Pollution"
        target = 2
        result = injector.inject_pollution(base, pollution, target, strategy='prepend')
        assert result.endswith(base)
        assert result.startswith("Pollution") # assuming 2 tokens covers "Pollution" which is 1 token
