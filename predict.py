import tensorflow as tf
import pandas as pd
import os
import numpy as np


class Rec_Model:
	def __init__(self, model_path, df, pred_count=1):
		self.model_path = model_path
		self.df = df
		self.pred_count = pred_count

	def make_predictions_df(self):
		model = tf.keras.models.load_model(self.model_path)
		df_predictions = pd.DataFrame()
		df_predictions['email_address'] = 'None'
		df_predictions['recommendation'] = 'None'
		df_features_for_pred = self.df.drop_duplicates(['email_address'])
		df_features_for_pred = df_features_for_pred.reset_index(drop=True)
		for row_index, row in df_features_for_pred.iterrows():
			df_slice = df_features_for_pred[df_features_for_pred['email_address'] == df_features_for_pred['email_address'][row_index]]
			df_slice_dict = {name: np.array(value) for name, value in df_slice.items()}
			_, preds = model((df_slice_dict))
			idx_num = preds[0, 0:self.pred_count]
			idx_num = tf.keras.backend.get_value(idx_num[0])
			df_predictions.loc[row_index, 'email_address'] = df_features_for_pred['email_address'][row_index]
			df_predictions.loc[row_index, 'recommendation'] = self.df['meeting_name'][idx_num]
		return df_predictions





