"""
FastAPI приложение для сервиса рекомендаций банковских продуктов.
"""
import os
import sys
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging

# --- Настройка логирования ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Пути к артефактам ---
# Добавляем корень проекта в sys.path для импорта модулей src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Импорты функций предобработки из вашего проекта
# (Предполагается, что эти функции не зависят от FastAPI напрямую)
from src.utils.preprocessing import handle_missing, encode_categorical
from src.features.engineering import generate_temporal_features, generate_aggregation_features

# --- Загрузка артефактов ---
MODEL_PATH = os.path.join(project_root, 'models', 'model.pkl')
FEATURES_PATH = os.path.join(project_root, 'models', 'feature_cols.pkl')

# Глобальные переменные для модели и списка признаков
model = None
feature_cols = None

def load_artifacts():
    """
    Функция для загрузки модели и списка признаков при запуске приложения.
    """
    global model, feature_cols
    if model is None or feature_cols is None:
        try:
            logger.info(f"Загрузка модели из {MODEL_PATH}")
            model = joblib.load(MODEL_PATH)
            logger.info(f"Загрузка списка признаков из {FEATURES_PATH}")
            feature_cols = joblib.load(FEATURES_PATH)
            logger.info(f"Артефакты загружены: Model={type(model).__name__}, Features={len(feature_cols)}")
        except FileNotFoundError as e:
            logger.error(f"Файл артефакта не найден: {e}")
            raise
        except Exception as e:
            logger.error(f"Ошибка при загрузке артефактов: {e}")
            raise

# --- Определение Pydantic моделей для валидации ---
class ClientInput(BaseModel):
    """
    Модель для валидации входных данных одного клиента.
    Включает основные признаки, используемые в модели.
    """
    ncodpers: Optional[int] = None
    age: Optional[float] = None
    renta: Optional[float] = None
    segmento: Optional[str] = None
    ind_empleado: Optional[str] = None
    sexo: Optional[str] = None
    canal_entrada: Optional[str] = None
    ind_nuevo: Optional[float] = None
    antiguedad: Optional[float] = None
    indrel: Optional[float] = None
    indrel_1mes: Optional[float] = None
    tiprel_1mes: Optional[str] = None
    indresi: Optional[str] = None
    indext: Optional[str] = None
    pais_residencia: Optional[str] = None
    indfall: Optional[str] = None
    tipodom: Optional[float] = None
    cod_prov: Optional[float] = None
    ind_actividad_cliente: Optional[float] = None
    # Добавьте другие признаки по мере необходимости

class PredictionRequest(BaseModel):
    """
    Модель для валидации запроса на предсказание.
    """
    clients: List[Dict[str, Any]]  # Используем Dict для гибкости, валидация внутри функции
    k: int = Field(default=3, ge=1, le=10)  # k от 1 до 10

class Recommendation(BaseModel):
    """
    Модель для одного элемента рекомендации.
    """
    product_index: int
    probability: float

class ClientRecommendation(BaseModel):
    """
    Модель для рекомендаций одного клиента.
    """
    client_id: int
    recommendations: List[Recommendation]

class PredictionResponse(BaseModel):
    """
    Модель для ответа на запрос предсказания.
    """
    recommendations: List[ClientRecommendation]

# --- Создание приложения FastAPI ---
app = FastAPI(
    title="Banking RecSys API",
    description="Сервис для рекомендации банковских продуктов.",
    version="1.0.0"
)

# --- Обработчики жизненного цикла ---
@app.on_event("startup")
async def startup_event():
    """
    Событие при запуске приложения.
    Используется для загрузки артефактов.
    """
    logger.info("Запуск приложения. Загрузка артефактов...")
    load_artifacts()
    logger.info("Артефакты загружены. Приложение готово к работе.")

# --- Эндпоинты ---
@app.get("/health")
async def health_check():
    """
    Эндпоинт для проверки состояния сервиса.
    """
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Эндпоинт для получения рекомендаций для списка клиентов.

    Args:
        request: Объект запроса, содержащий список клиентов и k (количество рекомендаций).

    Returns:
        Объект ответа с рекомендациями для каждого клиента.
    """
    if model is None or feature_cols is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    if not request.clients:
        # Возвращаем пустой список рекомендаций, если клиентов нет
        return PredictionResponse(recommendations=[])

    # 1. Преобразование входных данных в DataFrame
    try:
        df = pd.DataFrame(request.clients)
        logger.info(f"Получен запрос для {len(df)} клиентов.")
    except Exception as e:
        logger.error(f"Ошибка при создании DataFrame из входных данных: {e}")
        raise HTTPException(status_code=400, detail=f"Неверный формат данных клиентов: {e}")

    # 2. Предобработка данных (воссоздаём логику из ноутбука)
    # --- ВАЖНО: Убедитесь, что порядок и логика обработки совпадают с обучением ---
    # Фиксация типов для числовых колонок
    cols_to_fix_numeric = [
        'age', 'antiguedad', 'ind_nuevo', 'indrel', 'indrel_1mes',
        'tipodom', 'cod_prov', 'ind_actividad_cliente'
    ]
    for col in cols_to_fix_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Обработка пропусков (используем ту же стратегию, что и в ноутбуке)
    df = handle_missing(df, strategy='median')

    # Генерация признаков
    df = generate_temporal_features(df)
    # Для генерации агрегаций target_cols не требуется, так как мы на инференсе
    df = generate_aggregation_features(df, target_cols=[])

    # One-hot кодирование категориальных признаков
    # Список колонок для кодирования должен совпадать с обучением
    # (или быть подмножеством, если новых значений нет)
    cat_cols_to_encode = ['ind_empleado', 'sexo', 'segmento', 'canal_entrada', 'indresi',
                         'indext', 'pais_residencia', 'tiprel_1mes', 'indfall', 'product_tier']
    existing_cat_cols = [c for c in cat_cols_to_encode if c in df.columns]
    if existing_cat_cols:
        df = pd.get_dummies(df, columns=existing_cat_cols, drop_first=True, dtype=int)

    # Удаление служебных колонок
    cols_to_drop = ['ult_fec_cli_1t', 'nomprov', 'fecha_alta', 'fecha_dato']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # 3. Приведение к порядку признаков модели (Reindex)
    # Это критически важно для корректной работы sklearn-моделей
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0  # Добавляем недостающие колонки с нулевыми значениями
    df = df.reindex(columns=feature_cols, fill_value=0)

    # 4. Предсказание вероятностей
    try:
        y_pred_proba = model.predict_proba(df)
        logger.info(f"Предсказание выполнено для {len(df)} клиентов.")
    except Exception as e:
        logger.error(f"Ошибка при выполнении предсказания: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка модели: {e}")

    # 5. Формирование ответа (Топ-K)
    results = []
    for i, probs in enumerate(y_pred_proba):
        # Получаем индексы топ-K продуктов с наивысшей вероятностью
        top_k_indices = np.argsort(probs)[-request.k:][::-1]
        client_recs = []
        for idx in top_k_indices:
            client_recs.append(Recommendation(
                product_index=int(idx),
                probability=float(probs[idx])
            ))

        # Получаем ID клиента, если он был передан, иначе используем индекс
        client_id = request.clients[i].get("ncodpers", f"client_{i}")
        if isinstance(client_id, str):
            try:
                client_id = int(client_id)
            except ValueError:
                client_id = hash(client_id) # Используем хеш в крайнем случае

        results.append(ClientRecommendation(
            client_id=client_id,
            recommendations=client_recs
        ))

    return PredictionResponse(recommendations=results)