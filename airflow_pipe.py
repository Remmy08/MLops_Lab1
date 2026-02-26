import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from train_model2 import train

DATA_URL = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"

def download_data():
    df = pd.read_csv(DATA_URL, delimiter=',')
    df.to_csv("insurance.csv", index=False)
    print(f"df: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df

def clear_data():
    df = pd.read_csv("insurance.csv")
    
    cat_columns = ['sex', 'smoker', 'region']
    target_column = 'charges'
    
    df = df[(df['bmi'] >= 15) & (df['bmi'] <= 50)]
    df = df[(df['age'] >= 18) & (df['age'] <= 64)]
    
    q1 = df[target_column].quantile(0.01)
    q99 = df[target_column].quantile(0.99)
    df = df[(df[target_column] >= q1) & (df[target_column] <= q99)]
    
    ordinal = OrdinalEncoder()
    df[cat_columns] = ordinal.fit_transform(df[cat_columns])
    
    df.to_csv('df_clear.csv', index=False)
    print(f"Cleaned df: {df.shape}")
    return True

dag_insurance = DAG(
    dag_id="train_insurance_pipe",
    start_date=datetime(2025, 2, 3),
    schedule="@hourly",
    max_active_runs=1,
    catchup=False,
    default_args={'retries': 2}
)

download_task = PythonOperator(
    python_callable=download_data, 
    task_id="download_insurance", 
    dag=dag_insurance
)

clear_task = PythonOperator(
    python_callable=clear_data, 
    task_id="clear_insurance", 
    dag=dag_insurance
)

train_task = PythonOperator(
    python_callable=train, 
    task_id="train_insurance", 
    dag=dag_insurance,
    do_xcom_push=False
)

download_task >> clear_task >> train_task
