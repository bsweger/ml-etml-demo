"""
This script executes the DBSCAN clustering algorithm on the simulated taxi ride dataset.
"""

# from simulate_data import simulate_ride_data
import datetime
import logging

import boto3
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from mldemo.config.config import get_config
from mldemo.utils.extractor import Extractor

logging.basicConfig(level=logging.INFO)


class Clusterer:
    def __init__(
        self,
        bucket_name: str,
        bucket_prefix: str,
        file_name: str,
        model_params: dict = get_config().get('model_params', {'eps': 0.3, 'min_samples': 10}),
    ) -> None:
        self.model_params = model_params
        self.bucket_name = bucket_name
        self.bucket_prefix = bucket_prefix
        self.file_name = f'{bucket_prefix}/{file_name}'

    def cluster_and_label(self, features: list) -> None:
        logging.info({'msg': 'Extracting data from S3', 'object': f'{self.bucket_name}/{self.file_name}'})
        extractor = Extractor(self.bucket_name, self.file_name)
        df = extractor.extract_data()

        logging.info({'msg': 'Starting fit_transform'})
        df_features = df[features]
        df_features = StandardScaler().fit_transform(df_features)
        db = DBSCAN(**self.model_params).fit(df_features)

        # Find labels from the clustering
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Add labels to the dataset and return.
        df['label'] = labels

        date = datetime.datetime.now().strftime('%Y%m%d')
        object_key = f'{self.bucket_prefix}/clustered_data_{date}.json'
        boto3.client('s3').put_object(
            Body=df.to_json(orient='records'),
            Bucket=self.bucket_name,
            Key=object_key,
        )
        logging.info({'msg': 'Wrote clustered data to S3', 'object': object_key})
