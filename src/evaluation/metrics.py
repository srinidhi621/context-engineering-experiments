"""
Evaluation metrics for text generation.
"""
import re
from collections import Counter
from typing import List, Set

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """Check if the prediction is an exact match to the ground truth."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute the F1 score between prediction and ground truth."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def recall_score(prediction: str, ground_truth: str) -> float:
    """Compute the recall score."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if len(ground_truth_tokens) == 0:
        return 0
        
    return 1.0 * num_same / len(ground_truth_tokens)

def contains_score(prediction: str, ground_truth: str) -> bool:
    """Check if the ground truth is contained in the prediction."""
    return normalize_answer(ground_truth) in normalize_answer(prediction)