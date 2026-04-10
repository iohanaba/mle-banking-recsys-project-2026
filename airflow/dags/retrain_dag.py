from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib
import os
import sys
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

# Добавляем путь к проекту
sys.path.append('/home/mle-user/mle_projects/mle-practice-01/mle-banking-recsys-project-2026')

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': False,
}

dag = DAG(
    'retrain_recommendation_model',
    default_args=default_args,
    description='Pipeline for retraining banking recommendations',
    schedule_interval=timedelta(days=7),
    catchup=False,
    tags=['banking', 'recommendations'],
)

def extract_data(**kwargs):
    """Загрузка данных из CSV файлов"""
    data_path = 'data/raw/train_ver2.csv'
    
    if not os.path.exists(data_path):
        # Если файла нет, создаём тестовые данные
        print(f"File {data_path} not found. Creating sample data...")
        df = pd.DataFrame({
            'fecha_dato': pd.date_range('2015-01-01', periods=1000),
            'ncodpers': range(1000),
            'age': np.random.randint(18, 80, 1000),
            'renta': np.random.normal(25000, 10000, 1000),
            'segmento': np.random.choice(['01 - TOP', '02 - PARTICULARES', '03 - UNIVERSITARIO'], 1000)
        })
        # Добавляем таргет-колонки
        for i in range(5):
            df[f'ind_prod_{i}_ult1'] = np.random.choice([0, 1], 1000, p=[0.8, 0.2])
        
        os.makedirs('data/raw', exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"Sample data saved to {data_path}")
    else:
        # Загружаем реальные данные
        df = pd.read_csv(data_path, nrows=10000)  # Ограничиваем для скорости
        print(f"Loaded {len(df)} rows from {data_path}")
    
    # Сохраняем метаданные для следующих шагов
    ti = kwargs['ti']
    ti.xcom_push(key='data_shape', value=df.shape)
    ti.xcom_push(key='data_path', value=data_path)
    
    return data_path

def preprocess_data(**kwargs):
    """Предобработка данных"""
    ti = kwargs['ti']
    data_path = ti.xcom_pull(key='data_path', task_ids='extract_data')
    
    df = pd.read_csv(data_path, nrows=10000)
    
    # Базовая предобработка
    print("Starting preprocessing...")
    
    # Обработка пропусков
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Кодирование категориальных признаков
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('unknown')
            df[col] = df[col].astype('category').cat.codes
    
    # Сохраняем предобработанные данные
    processed_path = 'data/processed/processed_data.csv'
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv(processed_path, index=False)
    
    ti.xcom_push(key='processed_path', value=processed_path)
    print(f"Preprocessed data saved to {processed_path}")
    
    return processed_path

def train_model(**kwargs):
    """Обучение модели"""
    ti = kwargs['ti']
    processed_path = ti.xcom_pull(key='processed_path', task_ids='preprocess_data')
    
    df = pd.read_csv(processed_path)
    
    # Определяем признаки и таргеты
    exclude_cols = ['ncodpers', 'fecha_dato']
    target_cols = [col for col in df.columns if col.startswith('ind_') and col.endswith('_ult1')]
    feature_cols = [col for col in df.columns if col not in target_cols + exclude_cols]
    
    if len(target_cols) == 0:
        print("No target columns found. Creating dummy targets...")
        target_cols = ['ind_dummy_ult1']
        df[target_cols[0]] = np.random.choice([0, 1], len(df), p=[0.8, 0.2])
    
    X = df[feature_cols].fillna(0)
    y = df[target_cols]
    
    print(f"Training model with {X.shape[0]} samples and {X.shape[1]} features")
    print(f"Target columns: {target_cols}")
    
    # Разбиение на train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Обучение модели
    model = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, random_state=42),
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Сохранение модели
    os.makedirs('models', exist_ok=True)
    model_path = 'models/model_retrained.pkl'
    joblib.dump(model, model_path)
    
    # Сохранение feature_cols
    feature_cols_path = 'models/feature_cols_retrained.pkl'
    joblib.dump(feature_cols, feature_cols_path)
    
    ti.xcom_push(key='model_path', value=model_path)
    ti.xcom_push(key='feature_cols', value=feature_cols)
    
    print(f"Model saved to {model_path}")
    
    return model_path

def evaluate_model(**kwargs):
    """Оценка качества модели"""
    ti = kwargs['ti']
    model_path = ti.xcom_pull(key='model_path', task_ids='train_model')
    processed_path = ti.xcom_pull(key='processed_path', task_ids='preprocess_data')
    
    # Загружаем модель и данные
    model = joblib.load(model_path)
    df = pd.read_csv(processed_path)
    
    # Получаем feature_cols из предыдущего шага или вычисляем заново
    try:
        feature_cols = joblib.load('models/feature_cols_retrained.pkl')
    except:
        exclude_cols = ['ncodpers', 'fecha_dato']
        target_cols = [col for col in df.columns if col.startswith('ind_') and col.endswith('_ult1')]
        feature_cols = [col for col in df.columns if col not in target_cols + exclude_cols]
    
    target_cols = [col for col in df.columns if col.startswith('ind_') and col.endswith('_ult1')]
    
    if len(target_cols) == 0:
        print("No target columns for evaluation")
        return {}
    
    X = df[feature_cols].fillna(0)
    y = df[target_cols]
    
    # Предсказание
    y_pred_proba = model.predict_proba(X)
    
    # Расчёт метрик
    metrics = {}
    for i, col in enumerate(target_cols):
        if i < y_pred_proba.shape[1]:
            y_pred_binary = (y_pred_proba[:, i] > 0.5).astype(int)
            metrics[f'{col}_precision'] = precision_score(y[col], y_pred_binary, zero_division=0)
            metrics[f'{col}_recall'] = recall_score(y[col], y_pred_binary, zero_division=0)
    
    # Средняя точность и полнота
    metrics['avg_precision'] = np.mean([v for k, v in metrics.items() if 'precision' in k])
    metrics['avg_recall'] = np.mean([v for k, v in metrics.items() if 'recall' in k])
    
    # Логирование в MLflow
    try:
        mlflow.set_tracking_uri('http://localhost:5000')
        mlflow.set_experiment('banking-recsys-retrain')
        
        with mlflow.start_run(run_name=f'retrain_{datetime.now().strftime("%Y%m%d_%H%M%S")}'):
            mlflow.log_params({
                'model_type': 'OneVsRestClassifier_LogisticRegression',
                'retrain_date': datetime.now().isoformat(),
                'n_samples': len(df),
                'n_features': len(feature_cols)
            })
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, 'model')
            
            print(f"MLflow run completed. Metrics: {metrics}")
    except Exception as e:
        print(f"MLflow logging failed: {e}")
        print(f"Metrics calculated: {metrics}")
    
    # Сохранение метрик в файл
    import json
    metrics_path = 'models/retrain_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to {metrics_path}")
    
    return metrics

def register_model(**kwargs):
    """Регистрация модели как production-ready"""
    ti = kwargs['ti']
    metrics_path = 'models/retrain_metrics.json'
    
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Проверка качества модели
        avg_recall = metrics.get('avg_recall', 0)
        
        if avg_recall > 0.3:  # Порог качества
            print(f"Model quality is good (Recall: {avg_recall:.4f}). Registering as production model...")
            
            # Копируем модель как production
            import shutil
            shutil.copy('models/model_retrained.pkl', 'models/model.pkl')
            shutil.copy('models/feature_cols_retrained.pkl', 'models/feature_cols.pkl')
            
            print("Model registered successfully!")
        else:
            print(f"Model quality is below threshold (Recall: {avg_recall:.4f}). Not registering.")
            raise ValueError(f"Model recall {avg_recall:.4f} is below threshold 0.3")
    else:
        print("Metrics file not found. Skipping registration.")

# Определение задач
t1 = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    provide_context=True,
    dag=dag,
)

t2 = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    provide_context=True,
    dag=dag,
)

t3 = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    provide_context=True,
    dag=dag,
)

t4 = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    provide_context=True,
    dag=dag,
)

t5 = PythonOperator(
    task_id='register_model',
    python_callable=register_model,
    provide_context=True,
    dag=dag,
)

# Определение зависимостей
t1 >> t2 >> t3 >> t4 >> t5