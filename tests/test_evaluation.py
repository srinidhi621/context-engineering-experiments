"""Tests for evaluation metrics and judges"""
import pytest
from src.evaluation.metrics import exact_match_score, f1_score, contains_score
from src.evaluation.judges import MockJudge

def test_exact_match():
    assert exact_match_score("Hello World", "hello world") is True
    assert exact_match_score("Hello World.", "hello world") is True
    assert exact_match_score("Hello", "World") is False

def test_f1_score():
    assert f1_score("X Y Z", "X Y Z") == 1.0
    assert f1_score("X Y", "X Z") == 0.5  # 1 match out of 2 tokens each
    assert f1_score("X", "Y") == 0.0

def test_contains_score():
    assert contains_score("The answer is 42.", "42") is True
    assert contains_score("No answer here.", "42") is False

def test_mock_judge():
    judge = MockJudge()
    res = judge.evaluate("Q", "A", "Ref")
    assert res['score'] == 5
    assert res['label'] == "Mock"