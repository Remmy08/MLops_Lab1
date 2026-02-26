import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, PowerTransformer
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import root_mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from train_model import train

# URL нового датасета
DATA_URL = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"

def download_data():
    """Скачивает датасет страховых расходов"""
    df = pd.read_csv(DATA_URL, delimiter=',')
    df.to_csv("insurance.csv", index=False)
    print(f"df: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df

def clear_data():
    """Очищает и подготавливает данные"""
    df = pd.read_csv("insurance.csv")
    
    # Определение колонок
    cat_columns = ['sex', 'smoker', 'region']  # категориальные
    num_columns = ['age', 'bmi', 'children']    # числовые
    target_column = 'charges'                    # целевая переменная
    
    # Очистка от выбросов (здравый смысл + статистика)
    # Удаляем нереалистичные значения BMI
    df = df[(df['bmi'] >= 15) & (df['bmi'] <= 50)]
    
    # Удаляем нереалистичные значения возраста
    df = df[(df['age'] >= 18) & (df['age'] <= 64)]
    
    # Удаляем экстремальные значения charges (топ 1% и низ 1%)
    q1 = df['charges'].quantile(0.01)
    q99 = df['charges'].quantile(0.99)
    df = df[(df['charges'] >= q1) & (df['charges'] <= q99)]
    
    # Кодирование категориальных признаков
    ordinal = OrdinalEncoder()
    df[cat_columns] = ordinal.fit_transform(df[cat_columns])
    
    # Сохранение очищенных данных
    df.to_csv('df_clear.csv', index=False)
    print(f"Cleaned df: {df.shape}")
    return True

# Создание DAG
dag_cars = DAG(
    dag_id="train_insurance_pipe",
    start_date=datetime(2025, 2, 3),
    concurrency=4,
    schedule_interval="@hourly",
    max_active_runs=1,
    catchup=False,
    default_args={'retries': 2}
)

download_task = PythonOperator(
    python_callable=download_data, 
    task_id="download_insurance", 
    dag=dag_cars
)

clear_task = PythonOperator(
    python_callable=clear_data, 
    task_id="clear_insurance", 
    dag=dag_cars
)

train_task = PythonOperator(
    python_callable=train, 
    task_id="train_insurance", 
    dag=dag_cars
)

download_task >> clear_task >> train_task
