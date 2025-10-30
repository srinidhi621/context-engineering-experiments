"""Statistical analysis utilities"""
from scipy import stats
import numpy as np
from typing import Dict, Any

def paired_t_test(condition1: list, condition2: list) -> Dict[str, float]:
    """
    Paired t-test between two conditions
    
    Args:
        condition1: Scores for condition 1
        condition2: Scores for condition 2
        
    Returns:
        Dict with t_statistic and p_value
    """
    t_stat, p_value = stats.ttest_rel(condition1, condition2)
    return {'t_statistic': t_stat, 'p_value': p_value}

def cohen_d(group1: list, group2: list) -> float:
    """
    Cohen's d effect size
    
    Args:
        group1: Scores for group 1
        group2: Scores for group 2
        
    Returns:
        Cohen's d effect size
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

