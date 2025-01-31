#!/usr/bin/env python
# coding: utf-8

# # Final Project: Machine Learning System Development and Operation

# - Nama: Dzulfikri Adjmal
# - Email: dzulfikriadjmal@gmail.com
# - ID Dicoding: dzulfikriadjmal

# ## Import Library

# In[1]:


import pandas as pd
from typing import Text
from absl import logging

from tfx.orchestration import metadata, pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

from sklearn.preprocessing import LabelEncoder

import os
import shutil
from zipfile import ZipFile

import warnings

warnings.filterwarnings("ignore")


# ## Menyiapkan Dataset

# In[2]:


get_ipython().system('kaggle datasets download kushagra3204/sentiment-and-emotion-analysis-dataset')


# In[3]:


zip_file = "./sentiment-and-emotion-analysis-dataset.zip"
data_dir = "./data"
archive_dir = "./archive"
os.makedirs(data_dir, exist_ok=True)
with ZipFile(zip_file, "r") as zip_ref:
    zip_ref.extractall()


# In[4]:


list_data = os.listdir(archive_dir)

for file in list_data:
    shutil.move(os.path.join(archive_dir, file), os.path.join(data_dir, file))

os.removedirs(archive_dir)


# In[5]:


os.remove(os.path.join(data_dir, list_data[0]))
data_file = os.path.join(data_dir, list_data[1])


# In[6]:


df = pd.read_csv(data_file)
df.info()


# In[7]:


df.head()


# In[8]:


df["sentiment"].value_counts()


# In[9]:


import re


def clean_text(text: Text) -> Text:
    text = re.sub(r"([:;xX]-?[)DPOo3]|:[vV]|<3)", " ", text)
    text = re.sub(r"@[A-Za-z0-9]+", " ", text)
    text = re.sub(r"https?://[A-Za-z0-9./]+", " ", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"#\w+", " ", text)

    return text


df["sentence"] = df["sentence"].apply(clean_text)
df["sentence"] = df["sentence"].str.lower()


# In[10]:


le = LabelEncoder()
df["sentiment"] = le.fit_transform(df["sentiment"])


# In[11]:


df.sample(5)


# In[12]:


df.to_csv(data_file, index=False)


# ## Run Pipeline

# In[13]:


PIPELINE_NAME = "sentiment_pipeline"

DATA_ROOT = data_dir
TRANSFORM_MODULE_FILE = "./modules/sentence_sentiment_transform.py"
TRAINER_MODULE_FILE = "./modules/sentence_sentiment_trainer.py"

OUTPUT_ROOT = "output"
SERVING_MODEL_DIR = os.path.join(OUTPUT_ROOT, "serving_model")
PIPELINE_ROOT = os.path.join(OUTPUT_ROOT, PIPELINE_NAME)
METADATA = os.path.join(PIPELINE_ROOT, "metadata.sqlite")


def init_local_pipeline(
    components,
    pipeline_root: Text,
) -> pipeline.Pipeline:
    logging.info("Pipeline root set to: %s", pipeline_root)
    beam_args = [
        "--direct_running_mode=multi_processing",
        "--direct_num_workers=0",
    ]

    return pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(METADATA),
        eam_pipeline_args=beam_args,
    )


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)

    from modules.components import init_components

    component = init_components(
        data_dir=DATA_ROOT,
        transform_module=TRANSFORM_MODULE_FILE,
        training_module=TRAINER_MODULE_FILE,
        training_steps=500,
        eval_steps=200,
        serving_model_dir=SERVING_MODEL_DIR,
    )

    pipelines = init_local_pipeline(
        component,
        PIPELINE_ROOT,
    )
    BeamDagRunner().run(pipelines)


# In[ ]:




