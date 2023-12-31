# Extract, Transform, and Macine Learning Use Case (Demo)

This repo is a modified version of the Chapter 9 project in [_Machine Learing Engineering with Python_](https://bookshop.org/p/books/machine-learning-engineering-with-python-second-edition-manage-the-lifecycle-of-machine-learning-models-using-mlops-with-practical-examples-andrew-mcm/20564864?ean=9781837631964) by Andrew P. McMahon (second edition). The original code is available at https://github.com/PacktPublishing/Machine-Learning-Engineering-with-Python-Second-Edition.


## Background


## Overview


## Prerequisites

To run this code, you will need the following installed on your local machine:

* Python 3.10

In addition, you will need:

* Java installed on your system (required by Spark)
* An AWS S3 bucket + credentials that allow read/write access to it
* An OpenAI API key


## Setup

To run this project: clone the repo, open a terminal, make sure you're using Python 3.10, and follow the instructions below.

**Note:** These directions assume a macOS operating system and will likely require modification to work on a Windows machine.

### Install dependencies

1. From the root of the repo, create a virtual environment: `python3 -m venv .venv --prompt etml'
2. Activate the virtual environment: `source .venv/bin/activate`
3. Install the project as a local, editable package, and install its dependencies: `pip install -r requirements/requirements.txt -e .`

### Start Airflow

1. From the root of the repo, make sure you're in the virtual environment created above: `source .venv/bin/activate`
2. Initialize the local airflow installation: `airflow standalone`. The auto-generated `admin` password will be printed to the terminal
3. You should now be able to access the Airflow UI in your browser at http://localhost:8080. Log in with the username `admin` and the password printed to the terminal in the previous step