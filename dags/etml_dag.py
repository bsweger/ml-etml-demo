from __future__ import annotations

import datetime
import logging
import sys

import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator

from mldemo.config.config import get_config
from mldemo.utils import simulate
from mldemo.utils.cluster import Clusterer
from mldemo.utils.summarize import LLMSummarizer

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

bucket_name = get_config()['s3']['bucket_name']
bucket_prefix = get_config()['s3']['bucket_prefix']
date = datetime.datetime.now().strftime('%Y-%m-%d')
file_name = f'taxi-rides-{date}.json'
clustered_file_name = f'clustered_data_{date}.json'

with DAG(
    dag_id='etml_dag',
    start_date=pendulum.datetime(2021, 10, 1),
    schedule_interval='@daily',
    catchup=False,
) as dag:
    logging.info('DAG started')

    logging.info('Generating simulated data...')
    simulate_data_task = PythonOperator(
        task_id='simulate_data',
        python_callable=simulate.main,
    )

    logging.info('Extracting and clustering data...')
    extract_cluster_load_task = PythonOperator(
        task_id='extract_cluster_save',
        python_callable=Clusterer(bucket_name, bucket_prefix, file_name).cluster_and_label,
        op_kwargs={'features': ['ride_dist', 'ride_time']},
    )

    logging.info('Extracting and summarizing data...')
    extract_summarize_load_task = PythonOperator(
        task_id='extract_summarize',
        python_callable=LLMSummarizer(bucket_name, bucket_prefix, clustered_file_name).summarize,
    )

    simulate_data_task >> extract_cluster_load_task >> extract_summarize_load_task
