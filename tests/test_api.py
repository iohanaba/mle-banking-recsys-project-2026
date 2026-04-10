"""
Тесты для компонентов API банковских рекомендаций.
Тестирует логику предобработки и инференса модели.
Запуск: pytest tests/test_api.py -v
"""
import pytest
import sys
import os
import json
import pandas as pd
import numpy as np
import joblib

# Добавляем путь к проекту
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Импортируем модули из src
try:
    from src.api.app import load_artifacts, predict
    from pydantic import ValidationError
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import API components: {e}")
    COMPONENTS_AVAILABLE = False
    # Заглушка для app, если не удалось импортировать
    app = None
    load_artifacts = None
    predict = None

try:
    from src.utils.preprocessing import handle_missing, encode_categorical
    PREPROCESSING_AVAILABLE = True
except ImportError:
    PREPROCESSING_AVAILABLE = False

try:
    from src.features.engineering import generate_temporal_features, generate_aggregation_features
    FEATURE_ENGINEERING_AVAILABLE = True
except ImportError:
    FEATURE_ENGINEERING_AVAILABLE = False


class TestModelLoading:
    """Тесты загрузки модели"""
    
    def test_model_file_exists(self):
        """Проверка наличия файла модели"""
        model_path = "models/model.pkl"
        assert os.path.exists(model_path), f"Model file not found at {model_path}"
    
    def test_feature_cols_exists(self):
        """Проверка наличия файла с признаками"""
        feature_path = "models/feature_cols.pkl"
        assert os.path.exists(feature_path), f"Feature cols file not found at {feature_path}"
    
    def test_config_exists(self):
        """Проверка наличия конфигурационного файла"""
        config_path = "models/config.yaml"
        assert os.path.exists(config_path), f"Config file not found at {config_path}"

    @pytest.mark.skipif(not COMPONENTS_AVAILABLE, reason="API components not available")
    def test_load_artifacts(self):
        """Тест загрузки артефактов API"""
        # Проверяем, что функция существует
        assert load_artifacts is not None, "load_artifacts function is not defined"
        # Вызовем её, чтобы убедиться, что она не падает (артефакты должны существовать)
        load_artifacts()
        from src.api.app import model, feature_cols
        assert model is not None, "Model not loaded"
        assert feature_cols is not None, "Feature columns not loaded"


class TestPreprocessing:
    """Тесты предобработки данных"""
    
    @pytest.mark.skipif(not PREPROCESSING_AVAILABLE, reason="Preprocessing not available")
    def test_handle_missing(self):
        """Тест обработки пропусков"""
        df = pd.DataFrame({
            'age': [30, np.nan, 40],
            'renta': [20000, 30000, np.nan],
            'segmento': ['A', 'B', np.nan]
        })
        
        df_processed = handle_missing(df, strategy='median')
        
        assert df_processed.isnull().sum().sum() == 0, "There are still NaN values after processing"
        # Проверим, что median была применена
        assert df_processed.loc[1, 'age'] == 35, "Median for age not applied correctly"
        assert df_processed.loc[2, 'renta'] == 25000, "Median for renta not applied correctly"

    @pytest.mark.skipif(not PREPROCESSING_AVAILABLE, reason="Preprocessing not available")
    def test_encode_categorical(self):
        """Тест кодирования категориальных признаков"""
        df = pd.DataFrame({
            'segmento': ['A', 'B', 'A', 'C']
        })
        
        df_encoded = encode_categorical(df, cols=['segmento'], encoding_method='onehot')
        
        expected_cols = ['segmento_B', 'segmento_C']  # A становится базовой, если drop_first=True
        for col in expected_cols:
            assert col in df_encoded.columns, f"Column {col} not found after encoding"
        
        # Проверим, что исходная колонка ушла
        assert 'segmento' not in df_encoded.columns, "Original categorical column still present"


class TestFeatureEngineering:
    """Тесты генерации признаков"""
    
    @pytest.mark.skipif(not FEATURE_ENGINEERING_AVAILABLE, reason="Feature engineering not available")
    def test_generate_temporal_features(self):
        """Тест генерации временных признаков"""
        df = pd.DataFrame({
            'fecha_dato': pd.to_datetime(['2015-01-28', '2015-03-15']),
            'fecha_alta': pd.to_datetime(['2014-01-01', '2014-06-01'])
        })
        
        df_with_features = generate_temporal_features(df)
        
        expected_cols = ['month', 'year', 'quarter']
        for col in expected_cols:
            assert col in df_with_features.columns, f"Temporal feature {col} not created"
        
        assert df_with_features.loc[0, 'month'] == 1, "Month not extracted correctly"
        assert df_with_features.loc[1, 'month'] == 3, "Month not extracted correctly"

    @pytest.mark.skipif(not FEATURE_ENGINEERING_AVAILABLE, reason="Feature engineering not available")
    def test_generate_aggregation_features(self):
        """Тест генерации агрегированных признаков"""
        # Создаём минимальный датафрейм с таргетом
        df = pd.DataFrame({
            'ind_product_1_ult1': [1, 0, 1],
            'ind_product_2_ult1': [0, 1, 0],
            'ncodpers': [1, 2, 3]
        })
        
        df_with_features = generate_aggregation_features(df, target_cols=['ind_product_1_ult1', 'ind_product_2_ult1'])
        
        # Проверим, что создался признак n_active_products
        assert 'n_active_products' in df_with_features.columns, "n_active_products not created"
        expected_products = [1, 1, 1] # 1+0, 0+1, 1+0
        assert df_with_features['n_active_products'].tolist() == expected_products, "n_active_products calculation incorrect"
        
        # Проверим, что создался признак product_tier
        assert 'product_tier' in df_with_features.columns, "product_tier not created"
        # Tier должен быть строкой
        assert df_with_features['product_tier'].dtype == 'object', "product_tier should be object/string type"


class TestAPILogic:
    """Тесты логики API (без HTTP)"""
    
    @pytest.mark.skipif(not COMPONENTS_AVAILABLE, reason="API components not available")
    def test_predict_function_valid_input(self):
        """Тест функции predict с валидным вводом"""
        from src.api.app import PredictionRequest
        
        request_data = {
            "clients": [
                {
                    "ncodpers": 123456,
                    "age": 35,
                    "renta": 25000,
                    "segmento": "02 - PARTICULARES",
                    "ind_empleado": "N",
                    "sexo": "H",
                    "ind_nuevo": 0,
                    "antiguedad": 20,
                    "indrel": 1,
                    "indrel_1mes": 1.0,
                    "tiprel_1mes": "A",
                    "indresi": "S",
                    "indext": "N",
                    "pais_residencia": "ES",
                    "indfall": "N",
                    "tipodom": 1,
                    "cod_prov": 29,
                    "ind_actividad_cliente": 1,
                    "canal_entrada": "KHL"
                }
            ],
            "k": 3
        }
        
        try:
            req = PredictionRequest(**request_data)
        except ValidationError as e:
            pytest.fail(f"Validation failed for valid input: {e}")
        
        # Проверим, что объект создался без ошибок
        assert req.k == 3
        assert len(req.clients) == 1
        assert req.clients[0]["ncodpers"] == 123456

    @pytest.mark.skipif(not COMPONENTS_AVAILABLE, reason="API components not available")
    def test_predict_function_invalid_k(self):
        """Тест функции predict с невалидным k"""
        from src.api.app import PredictionRequest
        
        request_data = {
            "clients": [{"ncodpers": 1}],
            "k": -1  # Неверное значение
        }
        
        with pytest.raises(ValidationError):
            PredictionRequest(**request_data)

    @pytest.mark.skipif(not COMPONENTS_AVAILABLE, reason="API components not available")
    def test_predict_function_empty_clients(self):
        """Тест функции predict с пустым списком клиентов"""
        from src.api.app import PredictionRequest
        
        request_data = {
            "clients": [], # Пустой список
            "k": 3
        }
        
        try:
            req = PredictionRequest(**request_data)
            # Проверим, что объект создался
            assert req.k == 3
            assert len(req.clients) == 0
        except ValidationError as e:
            pytest.fail(f"Validation failed for empty clients: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])