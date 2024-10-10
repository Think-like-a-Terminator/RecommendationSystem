import tensorflow as tf
import tensorflow_recommenders as tfrs
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.callbacks import CSVLogger
from typing import Dict, Text
import time
import pandas as pd
import numpy as np
import os
import shutil
import datetime


class MeetingModel(tf.keras.Model):
	def __init__(self, unique_meeting_names, unique_architectures, unique_use_cases, unique_offer_names):
		super().__init__()
		self.unique_meeting_names = unique_meeting_names
		self.unique_architectures = unique_architectures
		self.unique_use_cases = unique_use_cases
		self.unique_offer_names = unique_offer_names
		
		self.meeting_name_embedding = tf.keras.Sequential([preprocessing.StringLookup(vocabulary=self.unique_meeting_names), 
														   layers.Embedding(len(self.unique_meeting_names)+2, 32)])
		self.architecture_embedding = tf.keras.Sequential([preprocessing.StringLookup(vocabulary=self.unique_architectures), 
														   layers.Embedding(len(self.unique_architectures)+2, 32)])
		self.use_cases_embedding = tf.keras.Sequential([preprocessing.StringLookup(vocabulary=self.unique_use_cases), 
														   layers.Embedding(len(self.unique_use_cases)+2, 32)])
		self.offer_names_embedding = tf.keras.Sequential([preprocessing.StringLookup(vocabulary=self.unique_offer_names), 
														   layers.Embedding(len(self.unique_offer_names)+2, 32)])
		
	def call(self, inputs):
		return tf.concat([self.meeting_name_embedding(inputs['meeting_name']),
						  self.architecture_embedding(inputs['architecture']), 
						  self.use_cases_embedding(inputs['use_case']),
						  self.offer_names_embedding(inputs['offer_name'])
						 ], axis = 1)


class UserModel(tf.keras.Model):
	def __init__(self, atx_data):
		super().__init__()
		self.atx_data = atx_data

		user_id_lookup_layer = tf.keras.layers.experimental.preprocessing.StringLookup(mask_token=None)
		user_id_lookup_layer.adapt(self.atx_data.map(lambda x: x['email_address']))
		user_id_embedding_dim = 32
		user_id_embedding_layer = tf.keras.layers.Embedding(input_dim=user_id_lookup_layer.vocabulary_size(), \
															output_dim=user_id_embedding_dim)
		self.user_embedding = tf.keras.Sequential([
			user_id_lookup_layer,
			user_id_embedding_layer,
		])
		
		job_title_lookup_layer = tf.keras.layers.experimental.preprocessing.StringLookup(mask_token=None)
		job_title_lookup_layer.adapt(self.atx_data.map(lambda x: x['job_title']))
		job_title_embedding_dim = 32
		job_title_embedding_layer = tf.keras.layers.Embedding(input_dim=job_title_lookup_layer.vocabulary_size(), \
															output_dim=job_title_embedding_dim)
		self.job_title_embedding = tf.keras.Sequential([
			job_title_lookup_layer,
			job_title_embedding_layer,
		])
		
		company_size_lookup_layer = tf.keras.layers.experimental.preprocessing.StringLookup(mask_token=None)
		company_size_lookup_layer.adapt(self.atx_data.map(lambda x: x['company_size']))
		company_size_embedding_dim = 32
		company_size_embedding_layer = tf.keras.layers.Embedding(input_dim=company_size_lookup_layer.vocabulary_size(), \
																output_dim=company_size_embedding_dim)
		self.company_size_embedding = tf.keras.Sequential([
			company_size_lookup_layer,
			company_size_embedding_layer,
		])

		country_lookup_layer = tf.keras.layers.experimental.preprocessing.StringLookup(mask_token=None)
		country_lookup_layer.adapt(self.atx_data.map(lambda x: x['country']))
		country_embedding_dim = 32
		country_embedding_layer = tf.keras.layers.Embedding(input_dim=country_lookup_layer.vocabulary_size(), \
															output_dim=country_embedding_dim)
		self.country_embedding = tf.keras.Sequential([
			country_lookup_layer,
			country_embedding_layer,
		])

	def call(self, inputs):
		return tf.concat([self.user_embedding(inputs["email_address"]), 
						  self.job_title_embedding(inputs["job_title"]), 
						  self.company_size_embedding(inputs["company_size"]), 
						  self.country_embedding(inputs['country'])], axis=1)


class RetrievalModel(tfrs.models.Model):
	def __init__(self, query_model, candidate_model, retrieval_task_layer):
		super().__init__()
		self.query_model: tf.keras.Model = query_model
		self.candidate_model: tf.keras.Model = candidate_model
		self.retrieval_task_layer: tf.keras.layers.Layer = retrieval_task_layer
 
	def compute_loss(self, features, training=False) -> tf.Tensor:
		query_embeddings = self.query_model(features)
		positive_candidate_embeddings = self.candidate_model(features)

		loss = self.retrieval_task_layer(
			query_embeddings,
			positive_candidate_embeddings
		)
		return loss


def map_data(atx_data):
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
	dict_data = {name: np.array(value) for name, value in df.items()}
	return dict_data


def create_tf_data(dict_data):
	tf_dataset_tensors = tf.data.Dataset.from_tensor_slices(dict_data)
	return tf_dataset_tensors

def create_candidate_data(df):
	df = df[['meeting_name', 'architecture', 'use_case', 'offer_name']]
	return df

def lr_sch_cb(epoch, lr):
	if epoch < 10:
		return lr
	else:
		return lr * 0.99


def train_model(df, tf_dataset, train_df, test_df, train_len, test_len, lr, num_epochs, model_dir, early_stop=False):
	'''
		train and save model
		hard coded optimizer Adagrad
	'''
	query_model = UserModel(tf_dataset)
	unique_meeting_names = np.unique(df['meeting_name'])
	unique_architectures = np.unique(df['architecture'])
	unique_use_cases = np.unique(df['use_case'])
	unique_offer_names = np.unique(df['offer_name']) 
	candidate_model = MeetingModel(unique_meeting_names, unique_architectures, unique_use_cases, unique_offer_names)
	candidates_dataset = tf_dataset.map(lambda x: {
		'meeting_name': x['meeting_name'], 
		'offer_name': x['offer_name'], 
		'architecture': x['architecture'], 
		'use_case': x['use_case'],
		})
	factorized_top_k_metrics = tfrs.metrics.FactorizedTopK(candidates=candidates_dataset.batch(128).map(candidate_model))
	retrieval_task_layer = tfrs.tasks.Retrieval(metrics=factorized_top_k_metrics)
	atx_retrieval_model = RetrievalModel(query_model, candidate_model, retrieval_task_layer)
	atx_retrieval_model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=lr))
	retrieval_atx_trainset = map_data(train_df)
	retrieval_atx_testset = map_data(test_df)
	retrieval_cached_atx_trainset = retrieval_atx_trainset.shuffle(train_len).batch(train_len).cache()
	retrieval_cached_atx_testset = retrieval_atx_testset.batch(test_len).cache()
	path = os.path.abspath(os.getcwd())
	ts = datetime.datetime.today().strftime('%Y_%m_%d_%H_%M')
	model_ts = 'model' + '_' + str(ts) 
	model_path = path + '\\' + model_dir + '\\' + model_ts
	chkpt_path = model_path + '\\' + 'callback_weights'
	if early_stop == True:
		callback = tf.keras.callbacks.ModelCheckpoint(chkpt_path, save_weights_only=True, monitor='loss', save_best_only=True)
		lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_sch_cb, verbose=1)
		history = atx_retrieval_model.fit(
			retrieval_cached_atx_trainset,
			validation_data=retrieval_cached_atx_testset,
			validation_freq=1,
			epochs=num_epochs,
			callbacks = [callback, lr_scheduler]
		)
		pd.DataFrame(history.history).to_csv(model_path + '\\' + 'callback_history_results.csv')
	else:
		csv_logger = CSVLogger('training_history_results.csv', separator=",", append=False)
		atx_retrieval_model.fit(
			retrieval_cached_atx_trainset,
			validation_data=retrieval_cached_atx_testset,
			validation_freq=1,
			epochs=num_epochs,
			callbacks = [csv_logger]
		)
	## Save model to directory
	recommendation_model = tfrs.layers.factorized_top_k.BruteForce(atx_retrieval_model.query_model)
	df_candidates = create_candidate_data(df)
	dict_data = create_dict_data(df_candidates)
	recommendation_model.index(candidate_model(dict_data))
	## passing one row of df into model to save initial input size for model
	df_features_for_pred = df.drop_duplicates(['email_address'])
	df_features_for_pred = df_features_for_pred.reset_index(drop=True)
	df_slice = df_features_for_pred[df_features_for_pred['email_address'] == df_features_for_pred['email_address'][1]]
	df_slice_dict = create_dict_data(df_slice)
	_, preds = recommendation_model((df_slice_dict))
	recommendation_model.save(model_path)
	if early_stop == False:
		results = model_path + '\\' + 'training_history_results.csv'
		shutil.move('training_history_results.csv', results)


def train_main(csvfile, model_dir, epochs, lr, early_stop=False, seed=55):
	path = os.path.abspath(os.getcwd())
	df = pd.read_csv(path + '\\' + 'data' + '\\' + csvfile)
	df = clean_data(df)
	dict_data = create_dict_data(df)
	tf_data = create_tf_data(dict_data)
	atx_tf_data = tf_data.map(lambda x: {
		"email_address": x["email_address"], 
		"job_title": x["job_title"],
		"company_size": x["company_size"],
		"country": x["country"],
		"meeting_name": x["meeting_name"],
		"offer_name": x["offer_name"],
		"architecture": x["architecture"],
		"use_case": x["use_case"]
		})
	trainset_size = 0.8 * atx_tf_data.__len__().numpy()
	atx_dataset_shuffled = atx_tf_data.shuffle(buffer_size=trainset_size, seed=seed, reshuffle_each_iteration=False)
	train_df = atx_dataset_shuffled.take(trainset_size)
	test_df = atx_dataset_shuffled.skip(trainset_size)
	testset_size = test_df.__len__()

	train_model(
		df=df, 
		tf_dataset=tf_data, 
		train_df=train_df, 
		test_df=test_df, 
		train_len=trainset_size, 
		test_len=testset_size,
		lr=lr,
		num_epochs=epochs, 
		model_dir =model_dir,
		early_stop=early_stop)
