from __future__ import annotations

import datetime
import pendulum
import os
import sys
from airflow import DAG
from airflow.operators.python import PythonOperator
from utils.summarize import LLMSummarizer
from utils.cluster import Clusterer
import logging


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

bucket_name = os.environ.get('S3_BUCKET_NAME', 'your-s3-bucket-name')
date = datetime.datetime.now().strftime('%Y-%m-%d')
file_name = f'taxi-rides-{date}.json'

with DAG(
    dag_ID = 'etml'dag',
    start_date=pendulum.datetime(2021, 10, 1),
    schedule_interval='@daily',
    catchup=False,
) as dag:
    logging.info('DAG started')
    logging.info('Extracting and clustering data...')
    extract_cluster_load_task = PythonOperator(
        task_id = 'extract_cluster_save',
        python_callable=Clusterer(bucket_name, file_name).\
            cluster_and_label,
        op_kwargs={'features': ['ride_dist', 'ride_time']}
    )

    logging.info('Extracting and summarizing data...')
    extract_summarize_load_task = PythonOperator(
        task_id='extract_summarize',
        python_callable=LLMSummarizer(bucket_name, file_name).summarize
    )

    extract_cluster_load_task >> extract_summarize_load_task
