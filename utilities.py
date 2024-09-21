import tensorflow as tf
from typing import Dict, Text
import time
import pandas as pd
import numpy as np
import os
import shutil
import datetime


def clean_data(df):
	'''
		Clean data and select the columns for input
	'''
	df = df.fillna('none')
	df = df.replace('[^a-zA-Z0-9]', '')
	df.columns = [x.lower() for x in df.columns]
	df = df[['meeting_name', 'offer_name', 'architecture', 'use_case', 
			'email_address', 'job_title', 'company_size', 'country']]
	for cols in df.columns:
		if df[cols].dtype == 'float64':
			df[cols] = df[cols].astype(np.float32)
		elif df[cols].dtype == 'int64' or df[cols].dtype == 'int32':
			df[cols] = df[cols].astype(np.float32)
	return df


def create_dict_data(df):
	'''
		Create dictionary from pandas dataframe
	'''
	dict_data = {name: np.array(value) for name, value in df.items()}
	return dict_data


def create_tf_data(dict_data):
	'''
		Create tf dataset from dictionary
	'''
	tf_dataset_tensors = tf.data.Dataset.from_tensor_slices(dict_data)
	return tf_dataset_tensors


def create_candidate_data(df):
	df = df[['meeting_name', 'architecture', 'use_case', 'offer_name']]
	return df


def map_data(atx_data):
	'''
		Must be tf dataset
	'''
	retrieval_data = atx_data.map(lambda x: {
			'email_address': x['email_address'],
			'job_title': x['job_title'],
			'company_size': x['company_size'],
			'country': x['country'], 
			'meeting_name': x['meeting_name'], 
			'offer_name': x['offer_name'], 
			'architecture': x['architecture'], 
			'use_case': x['use_case'],
		})
	return retrieval_data