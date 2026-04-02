"""
Модуль предобработки данных: обработка пропусков, кодирование, масштабирование.
"""
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def handle_missing(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    """
    Обработка пропущенных значений.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Исходный датафрейм
    strategy : str
        Стратегия импутации: 'median', 'mean', 'most_frequent', 'constant'
    
    Returns:
    --------
    pd.DataFrame с обработанными пропусками
    """
    df_copy = df.copy()
    
    # Числовые признаки: импутация медианой
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        imputer = SimpleImputer(strategy=strategy)
        df_copy[numeric_cols] = imputer.fit_transform(df_copy[numeric_cols])
    
    # Категориальные признаки: импутация модой
    categorical_cols = df_copy.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            if df_copy[col].isnull().any():
                mode_value = df_copy[col].mode()[0]
                df_copy[col] = df_copy[col].fillna(mode_value)
    
    return df_copy


def encode_categorical(
    df: pd.DataFrame,
    cols: list,
    encoding_method: str = 'onehot',
    drop_first: bool = True
) -> pd.DataFrame:
    """
    Кодирование категориальных признаков.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Исходный датафрейм
    cols : list
        Список колонок для кодирования
    encoding_method : str
        Метод кодирования: 'onehot' или 'label'
    drop_first : bool
        Удалять первый столбец при one-hot кодировании
    
    Returns:
    --------
    pd.DataFrame с закодированными признаками
    """
    df_copy = df.copy()
    
    if encoding_method == 'onehot':
        df_encoded = pd.get_dummies(
            df_copy,
            columns=cols,
            drop_first=drop_first,
            dtype=int
        )
        return df_encoded
    
    elif encoding_method == 'label':
        for col in cols:
            df_copy[col] = df_copy[col].astype('category').cat.codes
        return df_copy
    
    else:
        raise ValueError(f"Unknown encoding method: {encoding_method}")


def build_preprocessing_pipeline(
    numeric_features: list,
    categorical_features: list,
    numeric_strategy: str = 'median',
    categorical_strategy: str = 'most_frequent'
) -> ColumnTransformer:
    """
    Построение пайплайна предобработки для sklearn.
    
    Returns:
    --------
    ColumnTransformer готовый к fit/transform
    """
    numeric_transformer = StandardScaler()
    
    categorical_transformer = OneHotEncoder(
        handle_unknown='ignore',
        sparse_output=False,
        drop='first'
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    return preprocessor