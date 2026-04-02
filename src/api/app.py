"""
FastAPI application for Banking Product Recommendations.
"""
import os
import sys
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# Добавляем корень проекта в sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.preprocessing import handle_missing
from src.features.engineering import generate_temporal_features, generate_aggregation_features

app = FastAPI(title="Banking RecSys API", version="1.0.0")

MODEL_PATH = os.path.join(project_root, 'models', 'model.pkl')
FEATURES_PATH = os.path.join(project_root, 'models', 'feature_cols.pkl')

model = None
feature_cols = None

def load_artifacts():
    global model, feature_cols
    if model is None:
        try:
            model = joblib.load(MODEL_PATH)
            feature_cols = joblib.load(FEATURES_PATH)
            print(f"Artifacts loaded: Model={type(model).__name__}, Features={len(feature_cols)}")
        except Exception as e:
            print(f"Error loading artifacts: {e}")
            raise

@app.on_event("startup")
async def startup_event():
    load_artifacts()

class ClientInput(BaseModel):
    """Входные данные для одного клиента (упрощённая схема)"""
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

class PredictionRequest(BaseModel):
    clients: List[Dict[str, Any]]
    k: int = Field(default=3, ge=1, le=10)

class PredictionResponse(BaseModel):
    recommendations: List[Dict[str, Any]]

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None or feature_cols is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.clients:
        return {"recommendations": []}
    
    # 1. Преобразование входных данных в DataFrame
    # FIX: clients_data теперь внутри функции, где доступен request
    clients_data = [c if isinstance(c, dict) else {} for c in request.clients]
    df = pd.DataFrame(clients_data)
    
    # 2. Дефолтные значения для недостающих полей
    defaults = {
        'age': 30, 'renta': 20000, 'segmento': '02 - PARTICULARES',
        'ind_empleado': 'N', 'sexo': 'H', 'ind_nuevo': 0,
        'antiguedad': 12, 'indrel': 1, 'indrel_1mes': 1.0,
        'tiprel_1mes': 'A', 'indresi': 'S', 'indext': 'N',
        'pais_residencia': 'ES', 'indfall': 'N', 'tipodom': 1,
        'cod_prov': 28, 'ind_actividad_cliente': 1, 'canal_entrada': 'KHL'
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
    
    # 3. Предобработка
    df = handle_missing(df, strategy='median')
    
    # 4. Генерация признаков (с заглушками для инференса)
    df = generate_temporal_features(df)
    df = generate_aggregation_features(df, target_cols=[])  # target_cols пуст при инференсе
    
    # 5. One-hot encoding категориальных признаков
    cat_cols = ['ind_empleado', 'sexo', 'segmento', 'canal_entrada', 'indresi', 
                'indext', 'pais_residencia', 'tiprel_1mes', 'indfall', 'product_tier']
    existing_cats = [c for c in cat_cols if c in df.columns]
    if existing_cats:
        df = pd.get_dummies(df, columns=existing_cats, drop_first=True, dtype=int)
    
    # 6. Удаление служебных колонок
    cols_to_drop = ['ult_fec_cli_1t', 'nomprov', 'fecha_alta', 'fecha_dato']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    
    # 7. Приведение к порядку признаков модели (Reindex)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df.reindex(columns=feature_cols, fill_value=0)
    
    # 8. Предсказание
    try:
        y_pred_proba = model.predict_proba(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    # 9. Формирование ответа (Топ-K)
    results = []
    for i, probs in enumerate(y_pred_proba):
        top_k_indices = np.argsort(probs)[-request.k:][::-1]
        client_recs = []
        for idx in top_k_indices:
            client_recs.append({
                "product_index": int(idx),
                "probability": round(float(probs[idx]), 4)
            })
        
        client_id = request.clients[i].get("ncodpers", f"client_{i}")
        results.append({
            "client_id": client_id,
            "recommendations": client_recs
        })
    
    return {"recommendations": results}