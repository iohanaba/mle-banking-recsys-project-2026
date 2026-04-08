from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'retrain_recommendation_model',
    default_args=default_args,
    description='Pipeline for retraining banking recommendations',
    schedule_interval=timedelta(days=7),
    catchup=False,
)

def extract_data(**kwargs):
    print("Extracting data...")

def train_model(**kwargs):
    print("Training model...")

t1 = PythonOperator(task_id='extract', python_callable=extract_data, dag=dag)
t2 = PythonOperator(task_id='train', python_callable=train_model, dag=dag)

t1 >> t2