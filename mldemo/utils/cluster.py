'''
This script executes the DBSCAN clustering algorithm on the simulated taxi ride dataset.
'''

#from simulate_data import simulate_ride_data
import pandas as pd
import numpy as np
import datetime
import logging
logging.basicConfig(level=logging.INFO)

import boto3
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from mldemo.utils.extractor import Extractor

from mldemo.config.config import get_config

class Clusterer:
    def __init__(
        self, bucket_name: str, 
        file_name: str, 
        model_params: dict = get_config().get(
            'model_params',
            {'eps': 0.3, 'min_samples': 10})
    ) -> None:
        self.model_params = model_params
        self.bucket_name = bucket_name
        self.file_name = file_name
        
    def cluster_and_label(self, features: list) -> None:
        extractor = Extractor(self.bucket_name, self.file_name)
        df = extractor.extract_data()
        df_features = df[features]
        df_features = StandardScaler().fit_transform(df_features)
        db = DBSCAN(**self.model_params).fit(df_features)

        # Find labels from the clustering
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Add labels to the dataset and return.
        df['label'] = labels
        
        date = datetime.datetime.now().strftime("%Y%m%d")
        boto3.client('s3').put_object(
            Body=df.to_json(orient='records'), 
            Bucket=self.bucket_name, 
            Key=f"clustered_data_{date}.json"
        )
