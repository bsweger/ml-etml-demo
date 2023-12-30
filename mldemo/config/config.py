"""MVP config file."""

import os

#TODO: this is a pretty janky config, but it works

def get_config() -> dict:

    config = {}

    openai_model = 'gpt-3.5-turbo'

    model_params = {
        'eps': 0.3,
        'min_samples': 10,
    }

    s3 = {
        'bucket_name': os.environ.get('S3_BUCKET_NAME', 'your-s3-bucket-name'),
        'bucket_prefix': os.environ.get('S3_BUCKET_PREFIX', 'your-s3-bucket-prefix')
    }

    config['model_params'] = model_params
    config['openai_model'] = openai_model
    config['s3'] = s3

    return config
