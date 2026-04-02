"""
Модуль генерации признаков для рекомендательной системы.
"""
import pandas as pd
import numpy as np


def generate_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Генерация временных признаков из дата-колонок.
    
    Parameters:
    -----------
    df : pd.DataFrame с колонками fecha_dato, fecha_alta
    
    Returns:
    --------
    pd.DataFrame с добавленными временными признаками
    """
    df_copy = df.copy()
    
    # Извлечение компонентов даты
    if 'fecha_dato' in df_copy.columns:
        df_copy['fecha_dato'] = pd.to_datetime(df_copy['fecha_dato'])
        df_copy['month'] = df_copy['fecha_dato'].dt.month
        df_copy['year'] = df_copy['fecha_dato'].dt.year
        df_copy['quarter'] = df_copy['fecha_dato'].dt.quarter
    
    # Стаж клиента в месяцах на момент наблюдения
    if 'fecha_alta' in df_copy.columns and 'fecha_dato' in df_copy.columns:
        df_copy['fecha_alta'] = pd.to_datetime(df_copy['fecha_alta'], errors='coerce')
        df_copy['tenure_months'] = (
            (df_copy['fecha_dato'] - df_copy['fecha_alta']).dt.days / 30.44
        ).fillna(0).clip(lower=0)
    
    return df_copy


def generate_aggregation_features(
    df: pd.DataFrame,
    target_cols: list,
    group_col: str = 'ncodpers'
) -> pd.DataFrame:
    """
    Генерация агрегированных признаков по клиенту.
    
    Parameters:
    -----------
    df : pd.DataFrame
    target_cols : list колонок продуктов для агрегации
    group_col : колонка группировки (идентификатор клиента)
    
    Returns:
    --------
    pd.DataFrame с добавленными агрегациями
    """
    df_copy = df.copy()
    
    # Количество активных продуктов у клиента
    if len(target_cols) > 0:
        df_copy['n_active_products'] = df_copy[target_cols].sum(axis=1)
    
    # Сегментация по количеству продуктов
    df_copy['product_tier'] = pd.cut(
        df_copy['n_active_products'],
        bins=[-1, 0, 2, 5, 100],
        labels=['none', 'low', 'medium', 'high'],
        include_lowest=True
    )
    
    return df_copy


def generate_interaction_features(
    df: pd.DataFrame,
    col_pairs: list
) -> pd.DataFrame:
    """
    Генерация признаков-взаимодействий между колонками.
    
    Parameters:
    -----------
    df : pd.DataFrame
    col_pairs : list кортежей пар колонок для взаимодействия
    
    Returns:
    --------
    pd.DataFrame с добавленными признаками взаимодействий
    """
    df_copy = df.copy()
    
    for col1, col2 in col_pairs:
        if col1 in df_copy.columns and col2 in df_copy.columns:
            feature_name = f'{col1}_x_{col2}'
            # Для категориальных признаков создаём комбинацию
            if df_copy[col1].dtype == 'object' or df_copy[col2].dtype == 'object':
                df_copy[feature_name] = (
                    df_copy[col1].astype(str) + '_' + df_copy[col2].astype(str)
                )
            else:
                # Для числовых признаков — произведение
                df_copy[feature_name] = df_copy[col1] * df_copy[col2]
    
    return df_copy