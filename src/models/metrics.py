"""
Модуль метрик для оценки рекомендательной системы.
"""
import numpy as np
from typing import Union


def precision_at_k(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    k: int = 3
) -> float:
    """
    Расчёт Precision@K для multilabel-задачи.
    
    Parameters:
    -----------
    y_true : np.ndarray shape (n_samples, n_labels)
        Бинарная матрица истинных меток
    y_pred_proba : np.ndarray shape (n_samples, n_labels)
        Матрица предсказанных вероятностей
    k : int
        Размер топ-K рекомендаций
    
    Returns:
    --------
    float : среднее Precision@K по всем клиентам
    """
    n_samples = y_true.shape[0]
    precisions = []
    
    for i in range(n_samples):
        # Индексы топ-K предсказаний
        top_k_idx = np.argsort(y_pred_proba[i])[-k:][::-1]
        # Количество релевантных в топ-K
        relevant = y_true[i, top_k_idx].sum()
        # Precision для этого клиента
        precisions.append(relevant / k if k > 0 else 0.0)
    
    return np.mean(precisions)


def recall_at_k(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    k: int = 3
) -> float:
    """
    Расчёт Recall@K для multilabel-задачи.
    
    Returns:
    --------
    float : среднее Recall@K по всем клиентам
    """
    n_samples = y_true.shape[0]
    recalls = []
    
    for i in range(n_samples):
        top_k_idx = np.argsort(y_pred_proba[i])[-k:][::-1]
        relevant = y_true[i, top_k_idx].sum()
        total_relevant = y_true[i].sum()
        # Recall для этого клиента
        if total_relevant > 0:
            recalls.append(relevant / total_relevant)
        else:
            recalls.append(1.0)  # Нет релевантных — считаем идеальным
    
    return np.mean(recalls)


def map_at_k(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    k: int = 3
) -> float:
    """
    Расчёт Mean Average Precision@K.
    
    Returns:
    --------
    float : средний MAP@K по всем клиентам
    """
    n_samples = y_true.shape[0]
    ap_scores = []
    
    for i in range(n_samples):
        top_k_idx = np.argsort(y_pred_proba[i])[-k:][::-1]
        relevant = y_true[i, top_k_idx]
        
        if relevant.sum() == 0:
            ap_scores.append(0.0)
            continue
        
        # Расчёт Average Precision для одного клиента
        precision_sum = 0.0
        relevant_count = 0
        
        for rank, idx in enumerate(top_k_idx, start=1):
            if y_true[i, idx] == 1:
                relevant_count += 1
                precision_sum += relevant_count / rank
        
        ap = precision_sum / min(k, y_true[i].sum())
        ap_scores.append(ap)
    
    return np.mean(ap_scores)