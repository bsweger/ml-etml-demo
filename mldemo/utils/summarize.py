"""
This script will read in clustered taxi ride data from the clustered_data_{date}.json file and then 
use the OpenAI API to generate a summary of the text where the clustering returned a label of '-1' (i.e an outlier).

Once the summary is generated, it will be saved to a file called 'clustered_summarized_{date}.json' in the same AWS S3 bucket.

The textual data to be summarized is in the 'traffic', 'weather' and 'news' columns of the dataframe.

The prompt will be created using Langchain and will have the following format:

"
The following information describes conditions relevant to taxi journeys through a single day in Glasgow, Scotland.

News: {df['news'][i]}
Weather: {df['weather'][i]}
Traffic: {df['traffic'][i]}

Summarise the above information in 3 sentences or less.
"

The returned text will then be added to the pandas dataframe as df["summary"] and then saved to the clustered_summarized_{date}.json file in AWS S3.
"""

import datetime
import logging
import os
from textwrap import dedent

import boto3
import openai
from openai import OpenAI

from mldemo.config.config import get_config
from mldemo.utils.extractor import Extractor

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'], max_retries=2)


logging.basicConfig(level=logging.INFO)


class LLMSummarizer:
    def __init__(self, bucket_name: str, bucket_prefix: str, file_name: str) -> None:
        self.bucket_name = bucket_name
        self.bucket_prefix = bucket_prefix
        self.file_name = f'{bucket_prefix}/{file_name}'
        self.openai_model = get_config().get('openai_model', 'gpt-3.5-turbo')

    def summarize(self) -> None:
        logging.info({'msg': 'Extracting data from S3', 'object': f'{self.bucket_name}/{self.file_name}'})
        extractor = Extractor(self.bucket_name, self.file_name)
        df = extractor.extract_data()
        df['summary'] = ''

        logging.info({'msg': 'Adding prompts'})
        df['prompt'] = df.apply(lambda x: self.format_prompt(x['news'], x['weather'], x['traffic']), axis=1)

        logging.info({'msg': 'Attempting to summarize with OpenAI', 'model': self.openai_model})
        # for each row in the dataframe that represents an outlier in the clustered data, use an LLM
        # to summarize the record's news, weather, and traffic information
        df.loc[df['label'] == -1, 'summary'] = df.loc[df['label'] == -1, 'prompt'].apply(
            lambda x: self.generate_summary(x)
        )

        date = datetime.datetime.now().strftime('%Y%m%d')
        object_key = f'{self.bucket_prefix}/clustered_summarized_{date}.json'
        boto3.client('s3').put_object(
            Body=df.to_json(orient='records'),
            Bucket=self.bucket_name,
            Key=object_key,
        )
        logging.info({'msg': 'Wrote summarized data to S3', 'object': object_key})

    def format_prompt(self, news: str, weather: str, traffic: str) -> str:
        prompt = dedent(
            f"""
            The following information describes conditions relevant to taxi journeys through a single day in Glasgow, Scotland.

            News: {news}
            Weather: {weather}
            Traffic: {traffic}

            Summarise the above information in 3 sentences or less.
            """
        )
        return prompt

    def generate_summary(self, prompt: str) -> str:
        # Try the chatgpt model and fall back to another one if necessary
        try:
            response = client.chat.completions.create(
                model=self.openai_model, temperature=0.3, messages=[{'role': 'user', 'content': prompt}]
            )
            return response.choices[0].message.content
        except Exception:
            fallback_model = 'text-davinci-003'
            logging.info({'msg': 'Attempting fallback chatgpt summary', 'mode': fallback_model})
            try:
                response = client.completions.create(model=fallback_model, prompt=prompt)
                return response.choices[0].text
            except openai.RateLimitError as e:
                # For 429s, Openai returns header info with more specifics
                # https://platform.openai.com/docs/guides/rate-limits/rate-limits-in-headers
                # These headers are supposed to be available via the Python client, though I
                # couldn't see them when testing. Logging the response headers anyway.
                # https://github.com/openai/openai-python/issues/416#issuecomment-1795428669
                logging.exception({'msg': e.message, 'response_headers': e.response.headers})
                # If we're getting rate limited, just stop the demo
                raise e
