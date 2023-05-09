import sys
from airflow import DAG
from airflow.contrib.operators.spark_submit_operator import SparkSubmitOperator
from datetime import datetime
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
sys.path.append('/home/ebony_maw_0/code')
from Model_Training_function import train_model

dag = DAG(
            'Raw_To_Processed_Job',
            description='Job to process raw files',
            start_date = datetime(2023,1,1),
            schedule_interval='@daily'
        )

spark_submit = SparkSubmitOperator(
            task_id='Raw_to_Processed',
            conn_id='spark_default',
            application='/home/ebony_maw_0/code/Raw_To_Processed.py',
            verbose=True,
            conf={
                    'spark.driver.memory':'15g',
                    'spark.executor.memory':'40g',
                    'spark.master':'local'
                },
            dag=dag
        )

spark_submit2 = SparkSubmitOperator(
            task_id='Feature_Engineering',
            conn_id='spark_default',
            application='/home/ebony_maw_0/code/Feature_Engineering.py',
            verbose=True,
            conf={
                    'spark.driver.memory':'15g',
                    'spark.executor.memory':'40g',
                    'spark.master':'local'
                },
            dag=dag
        )

file_rename = BashOperator(
            task_id='Rename_file',
            bash_command='mv /home/ebony_maw_0/data/model_training_data/*.csv /home/ebony_maw_0/data/model_training_data/training_data.csv',
            dag=dag
        )

model_training = PythonOperator(
            task_id='Model_Training',
            python_callable=train_model,
            dag=dag
        )

spark_submit >> spark_submit2 >>file_rename >> model_training
