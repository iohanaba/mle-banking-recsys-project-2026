"""
Модуль мониторинга: сбор и отправка метрик качества, drift и latency.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# --- Настройка логгера ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- Настройка файла логов ---
LOG_DIR = Path('logs')
LOG_DIR.mkdir(exist_ok=True)
METRICS_LOG = LOG_DIR / 'metrics.jsonl'

# --- Настройка обработчика ---
handler = logging.FileHandler(METRICS_LOG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# --- Убедиться, что обработчик не дублируется при повторном импорте ---
if not logger.handlers:
    logger.addHandler(handler)

# --- Telegram Bot Integration (если настроен) ---
TELEGRAM_BOT_TOKEN = None  # Установите токен из .env или конфига
TELEGRAM_CHAT_ID = None    # Установите chat_id из .env или конфига

try:
    # Попытка импорта requests для отправки в Telegram
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    logger.warning("requests library not found. Telegram alerts disabled.")


def send_telegram_alert(message: str):
    """
    Отправка алерта в Telegram.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID or not HAS_REQUESTS:
        logger.debug("Telegram alert skipped (token/chat_id not set or requests not available)")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        logger.info(f"Telegram alert sent: {message[:50]}...")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send Telegram alert: {e}")


def log_metric(name: str, value: float, tags: Optional[Dict[str, Any]] = None, timestamp: Optional[str] = None, run_id: Optional[str] = None):
    """
    Логирование метрики в JSONL-файл.

    Parameters:
    -----------
    name : str
        Название метрики (например, 'precision_at_3')
    value : float
        Значение метрики
    tags : dict, optional
        Дополнительные теги (run_id, model_version, etc.)
    timestamp : str, optional
        Временная метка в ISO-формате
    run_id : str, optional
        ID эксперимента MLflow (если доступен)
    """
    entry = {
        'timestamp': timestamp or datetime.utcnow().isoformat(),
        'metric': name,
        'value': value,
        'tags': tags or {}
    }
    if run_id:
        entry['tags']['run_id'] = run_id

    with open(METRICS_LOG, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    logger.info(f"Logged metric: {name}={value}")


def calculate_psi(expected: List[float], actual: List[float], bins: int = 10) -> float:
    """
    Расчёт Population Stability Index для мониторинга drift.

    Parameters:
    -----------
    expected : list
        Распределение признака на тренировочных данных (источник истины)
    actual : list
        Распределение признака на текущих данных (production)
    bins : int
        Количество бинов для гистограммы

    Returns:
    --------
    float : PSI значение (>0.25 — значительный дрейф)
    """
    # Конвертируем в numpy для удобства
    expected = np.array(expected)
    actual = np.array(actual)

    # Удаляем NaN перед расчётом
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    # Проверка на пустые массивы
    if len(expected) == 0 or len(actual) == 0:
        logger.warning("Cannot calculate PSI: one of the distributions is empty or contains only NaN.")
        return np.nan

    # Границы бинов по ожидаемому (базовому) распределению
    bin_edges = np.histogram_bin_edges(expected, bins=bins)

    # Гистограммы частот (не плотности!)
    exp_hist, _ = np.histogram(expected, bins=bin_edges)
    act_hist, _ = np.histogram(actual, bins=bin_edges)

    # Нормализация (частоты -> доли)
    # Добавляем маленькое значение (smoothing) для избежания деления на 0
    smoothing = 0.0001
    exp_pct = (exp_hist + smoothing) / (exp_hist.sum() + smoothing * bins)
    act_pct = (act_hist + smoothing) / (act_hist.sum() + smoothing * bins)

    # PSI формула: sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct))
    # np.where для безопасного вычисления log(0), заменяя его на 0
    log_ratio = np.where(
        (act_pct > 0) & (exp_pct > 0),
        np.log(act_pct / exp_pct),
        0
    )
    psi = np.sum((act_pct - exp_pct) * log_ratio)

    return float(psi)


def log_drift_report(feature_name: str, psi_value: float, threshold: float = 0.1):
    """
    Логирование отчёта о дрейфе признака и отправка алерта при превышении порога.
    """
    if np.isnan(psi_value):
        status = 'INVALID'
        message = f"PSI for {feature_name} is NaN (likely due to empty/NaN data)."
        logger.warning(message)
    else:
        status = 'WARNING' if psi_value > threshold else 'OK'
        message = f"Drift check for {feature_name}: PSI={psi_value:.4f} [{status}] (threshold: {threshold})"

    log_metric(
        name=f'drift_psi_{feature_name}',
        value=psi_value,
        tags={'status': status, 'threshold': threshold}
    )

    logger.info(message)

    if status == 'WARNING':
        alert_msg = f"DRIFT ALERT: PSI for '{feature_name}' is {psi_value:.4f} > {threshold}!"
        send_telegram_alert(alert_msg)


def precision_at_k(y_true: np.ndarray, y_pred_proba: np.ndarray, k: int = 3) -> float:
    """
    Расчёт Precision@K для multilabel-задачи.
    y_true: бинарная матрица (n_samples, n_labels)
    y_pred_proba: матрица вероятностей (n_samples, n_labels)
    """
    n_samples, n_labels = y_true.shape
    if k > n_labels:
        k = n_labels
        logger.warning(f"k ({k}) was greater than n_labels ({n_labels}), adjusted to n_labels.")

    # Получаем индексы топ-k вероятностей для каждого сэмпла
    top_k_indices = np.argpartition(y_pred_proba, -k, axis=1)[:, -k:]

    # Создаём бинарную матрицу для топ-k предсказаний
    y_pred_top_k = np.zeros_like(y_pred_proba, dtype=bool)
    # Используем np.arange для индексации по строкам
    y_pred_top_k[np.arange(n_samples)[:, None], top_k_indices] = True

    # Пересечение предсказанных и истинных меток
    intersection = (y_true.astype(bool) & y_pred_top_k).sum(axis=1)

    # Precision@K: сумма пересечений / k
    # np.clip для защиты от деления на 0 (если k=0, хотя вряд ли)
    precisions = intersection / np.clip(k, a_min=1, a_max=None)

    return float(precisions.mean())


def recall_at_k(y_true: np.ndarray, y_pred_proba: np.ndarray, k: int = 3) -> float:
    """
    Расчёт Recall@K для multilabel-задачи.
    y_true: бинарная матрица (n_samples, n_labels)
    y_pred_proba: матрица вероятностей (n_samples, n_labels)
    """
    n_samples, n_labels = y_true.shape
    if k > n_labels:
        k = n_labels
        logger.warning(f"k ({k}) was greater than n_labels ({n_labels}), adjusted to n_labels.")

    # Получаем индексы топ-k вероятностей для каждого сэмпла
    top_k_indices = np.argpartition(y_pred_proba, -k, axis=1)[:, -k:]

    # Создаём бинарную матрицу для топ-k предсказаний
    y_pred_top_k = np.zeros_like(y_pred_proba, dtype=bool)
    y_pred_top_k[np.arange(n_samples)[:, None], top_k_indices] = True

    # Пересечение предсказанных и истинных меток
    intersection = (y_true.astype(bool) & y_pred_top_k).sum(axis=1)

    # Общее количество истинных меток для каждого сэмпла
    total_relevant = y_true.sum(axis=1)

    # Recall@K: пересечение / общее количество истинных
    # Защита от деления на 0: если total_relevant = 0, то recall = 1.0 (или 0.0, зависит от интерпретации)
    # Здесь используется 1.0, так как если у сэмпла нет релевантных меток, и мы ничего не предсказали, это идеальный recall.
    recalls = np.divide(intersection, total_relevant, out=np.ones_like(intersection, dtype=float), where=total_relevant!=0)

    return float(recalls.mean())


def average_precision_at_k(y_true: np.ndarray, y_pred_proba: np.ndarray, k: int = 3) -> float:
    """
    Вспомогательная функция для расчёта AP@K для одного сэмпла.
    Возвращает среднюю AP@K по всем сэмплам.
    """
    n_samples, n_labels = y_true.shape
    if k > n_labels:
        k = n_labels
        logger.warning(f"k ({k}) was greater than n_labels ({n_labels}), adjusted to n_labels.")

    aps = []
    for i in range(n_samples):
        y_true_i = y_true[i]
        y_pred_i = y_pred_proba[i]

        # Индексы топ-k предсказаний (по убыванию)
        top_k_idx = np.argsort(y_pred_i)[::-1][:k]

        score = 0.0
        num_hits = 0.0
        for j, idx in enumerate(top_k_idx):
            if y_true_i[idx] == 1:
                num_hits += 1.0
                # Precision на шаге j (от 1 до k)
                precision_at_j = num_hits / (j + 1)
                score += precision_at_j

        # Нормализуем на min(k, total_relevant_for_this_sample)
        total_rel = y_true_i.sum()
        if total_rel > 0:
            aps.append(score / min(k, total_rel))
        else:
            # Если у сэмпла нет релевантных меток, AP = 0 (или 1, если не было true positives)
            # Принято считать AP = 0 в таком случае
            aps.append(0.0)

    return float(np.mean(aps))


def map_at_k(y_true: np.ndarray, y_pred_proba: np.ndarray, k: int = 3) -> float:
    """
    Расчёт Mean Average Precision@K для multilabel-задачи.
    y_true: бинарная матрица (n_samples, n_labels)
    y_pred_proba: матрица вероятностей (n_samples, n_labels)
    """
    return average_precision_at_k(y_true, y_pred_proba, k)


def log_quality_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, k: int = 3, run_id: Optional[str] = None):
    """
    Логирование основных метрик качества модели.
    """
    p_at_k = precision_at_k(y_true, y_pred_proba, k=k)
    r_at_k = recall_at_k(y_true, y_pred_proba, k=k)
    map_at_k_val = map_at_k(y_true, y_pred_proba, k=k)

    try:
        # ROC-AUC считается по каждому лейблу, затем усредняется (macro)
        auc_macro = roc_auc_score(y_true, y_pred_proba, average='macro', multi_class='ovr')
    except ValueError as e:
        logger.warning(f"Could not calculate ROC-AUC: {e}")
        auc_macro = np.nan

    log_metric('precision_at_k', p_at_k, tags={'k': k}, run_id=run_id)
    log_metric('recall_at_k', r_at_k, tags={'k': k}, run_id=run_id)
    log_metric('map_at_k', map_at_k_val, tags={'k': k}, run_id=run_id)
    log_metric('roc_auc_macro', auc_macro, tags={'average': 'macro'}, run_id=run_id)

    logger.info(f"Quality metrics logged for k={k}: P@{k}={p_at_k:.4f}, R@{k}={r_at_k:.4f}, MAP@{k}={map_at_k_val:.4f}, AUC={auc_macro:.4f}")


def log_prediction_distribution(y_pred_proba: np.ndarray, feature_name: str = "prediction_probability"):
    """
    Логирование статистик по распределению предсказаний (для мониторинга деградации).
    """
    mean_prob = float(np.mean(y_pred_proba))
    std_prob = float(np.std(y_pred_proba))
    min_prob = float(np.min(y_pred_proba))
    max_prob = float(np.max(y_pred_proba))

    log_metric(f'dist_mean_{feature_name}', mean_prob)
    log_metric(f'dist_std_{feature_name}', std_prob)
    log_metric(f'dist_min_{feature_name}', min_prob)
    log_metric(f'dist_max_{feature_name}', max_prob)

    logger.info(f"Prediction distribution stats logged for {feature_name}: mean={mean_prob:.4f}, std={std_prob:.4f}")


# --- Пример использования ---
if __name__ == "__main__":
    # Пример генерации тестовых данных
    np.random.seed(42)
    n_samples, n_labels = 100, 24
    y_true_example = np.random.choice([0, 1], size=(n_samples, n_labels), p=[0.9, 0.1])
    y_pred_proba_example = np.random.rand(n_samples, n_labels)

    # Логирование метрик качества
    log_quality_metrics(y_true_example, y_pred_proba_example, k=3)

    # Пример логирования дрейфа (сравниваем разные "пакеты" данных)
    feature_train = np.random.normal(loc=0, scale=1, size=1000)
    feature_current = np.random.normal(loc=0.1, scale=1.1, size=100) # Легкий дрейф
    psi_val = calculate_psi(feature_train, feature_current)
    log_drift_report('example_feature', psi_val)

    # Пример логирования распределения предсказаний
    log_prediction_distribution(y_pred_proba_example.flatten())

    print("Example metrics logged to logs/metrics.jsonl")