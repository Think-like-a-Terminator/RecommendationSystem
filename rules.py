import pandas as pd
import logging

class Rules:
	def __init__(self, df, cxcustomerbuid_col=None, emailaddress_col=None, stage_col=None, usecase_col=None, daysinstage_col=None, \
					sessionname_col=None, sessiondate_col=None):
		self.df = df
		self.cxcustomerbuid_col = cxcustomerbuid_col
		self.emailaddress_col = emailaddress_col
		self.stage_col = stage_col
		self.usecase_col = usecase_col
		self.daysinstage_col = daysinstage_col
		self.sessionname_col = sessionname_col
		self.sessiondate_col = sessiondate_col

	def get_top_stages(self, account_level=False):
		'''
			get aggregated stages count for all individuals or by company/account level

			param account_level:  set to True if wanting stages count by account level otherwise returns aggregated count from all individuals/records

			returns df
		'''
		if account_level == False:
			if bool(self.stage_col) == False:
				msg = 'Expected stage_col param to be set, but did not set stage_col when initializing rules class'
				logging.error(msg)
				raise ParamUnavailable(msg)
			stage_count = self.df.groupby([self.stage_col])[self.stage_col].count().reset_index(name="stage_count")
			stage_count = stage_count.sort_values(by='stage_count', ascending=False)
			stage_count = stage_count.reset_index(drop=True)
			return stage_count
		else:
			if bool(self.stage_col) == False or bool(self.cxcustomerbuid_col) == False:
				msg = 'Expected stage_col or cxcustomerbuid_col param to be set, but did not set one or both of them \
						when initializing rules class'
				logging.error(msg)
				raise ParamUnavailable(msg)
			stage_count = self.df.groupby([self.cxcustomerbuid_col, self.stage_col])[self.stage_col].count().reset_index(name="stage_count")
			stage_count = stage_count.sort_values(by=[self.cxcustomerbuid_col, 'stage_count'], ascending=False)
			stage_count = stage_count.reset_index(drop=True)
			return stage_count


	def get_top_use_cases(self, account_level=False):
		'''
			get aggregated use_case count for all companies or by company/account level and returns df

			param account_level:  set to True if wanting use_case count by account level otherwise returns aggregated count from all accounts

			returns df
		'''
		if account_level == False:
			if bool(self.usecase_col) == False:
				msg = 'Expected usecase_col param to be set, but did not set usecase_col when initializing rules class'
				logging.error(msg)
				raise ParamUnavailable(msg)
			usecase_count = self.df.groupby([self.stage_col, self.usecase_col])[self.usecase_col].count().reset_index(name="usecase_count")
			usecase_count = usecase_count.sort_values(by=[self.stage_col, 'usecase_count'], ascending=False)
			usecase_count = usecase_count.reset_index(drop=True)
			return usecase_count
		else:
			if bool(self.usecase_col) == False or bool(self.cxcustomerbuid_col) == False:
				msg = 'Expected usecase_col or cxcustomerbuid_col param to be set, but did not set one or both of them \
						when initializing rules class'
				logging.error(msg)
				raise ParamUnavailable(msg)
			usecase_count = self.df.groupby([self.cxcustomerbuid_col, self.usecase_col])[self.usecase_col].count().reset_index(name="usecase_count")
			usecase_count = usecase_count.sort_values(by=[self.cxcustomerbuid_col, 'usecase_count'], ascending=False)
			usecase_count = usecase_count.reset_index(drop=True)
			return usecase_count


	def remove_sessions(self, lookback_date=None, sessions_to_remove=None, stages_to_remove=None):
		'''
			At least one of these params need to be set:  sessions_to_remove or stages_to_remove

			param: sessions_to_remove:  removes any ATX sessions user specifies using session/meeting names 
			
			param: stages_to_remove: removes any ATX sessions user specificies using stages

			param: Lookback_date:  filter df starting from lookback_date if given
			
			sessions_to_remove or stages_to_remove params can be single value or list of values

			returns df
		'''
		if bool(lookback_date) == True:
			if bool(self.sessiondate_col) == False:
				msg = 'Expected sessiondate_col to be set, but did not set sessiondate_col when initializing rules class'
				logging.error(msg)
				raise ParamUnavailable(msg)
			else:
				recent_df = pd.to_datetime(self.df[self.sessiondate_col], errors='coerce')
				recent_df = recent_df[recent_df[self.sessiondate_col] >= lookback_date]
				recent_df = recent_df.reset_index(drop=True)
		else:
			recent_df = self.df
		if bool(sessions_to_remove) == True:
			if bool(self.sessionname_col) == False:
				msg = 'Expected sessionname_col to be set, but did not set sessionname_col when initializing rules class'
				logging.error(msg)
				raise ParamUnavailable(msg)
			else:
				if isinstance(sessions_to_remove, list) == True: 
					recent_df = recent_df[~recent_df[self.sessionname_col].isin(sessions_to_remove)]
					recent_df = recent_df.reset_index(drop=True)
					return recent_df
				else:
					recent_df = recent_df[recent_df[self.sessionname_col] != sessions_to_remove]
					recent_df = recent_df.reset_index(drop=True)
					return recent_df
		elif bool(stages_to_remove) == True:
			if isinstance(stages_to_remove, list) == True: 
				recent_df = recent_df[~recent_df[self.stage_col].isin(stages_to_remove)]
				recent_df = recent_df.reset_index(drop=True)
				return recent_df
			else:
				recent_df = recent_df[recent_df[self.stage_col] != stages_to_remove]
				recent_df = recent_df.reset_index(drop=True)
				return recent_df
		else:
			msg = 'Did not set sessions_to_remove param or stages_to_remove param, need to set one of these params'
			logging.error(msg)
			raise ParamUnavailable(msg)


	def remove_prior_attended_sessions(self, df_predictions):
		'''
			removes recommendations from model predictions df where the individual already attended the session in the past if any

			set df in Rules class param to historical ATX attended sessions

			param df_predictions:   dataframe of the recommendations from the model

			returns new df of predictions 
		'''
		if bool(self.sessionname_col) == False or bool(self.emailaddress_col) == False:
			msg = 'Expected sessionname_col and emailaddress_col to be set, but did not set one or both of them when initializing rules class'
			logging.error(msg)
			raise ParamUnavailable(msg)
		elif 'email_address' not in df_predictions.columns and 'recommendation' not in df_predictions.columns:
			msg = 'Predictions dataframe does not have email_address and recommendation columns or requires column names to be named so'
			logging.error(msg)
			raise ColumnsUnavailable(msg)
		else:
			df_history = self.df[[self.emailaddress_col, self.sessionname_col]]
			df_history = df_history.drop_duplicates(subset=[self.emailaddress_col, self.sessionname_col])
			df_history = df_history.reset_index(drop=True)
			df_history = df_history.rename(columns={self.emailaddress_col: 'email_address', self.sessionname_col: 'meeting_name'})
			df_predictions = df_predictions.rename(columns={'recommendation': 'meeting_name'})
			combined = pd.merge(df_predictions, df_history, on=['email_address', 'meeting_name'], how='outer', indicator=True)
			predictions = combined.loc[combined["_merge"] == "left_only"].drop("_merge", axis=1)
			predictions = predictions.rename(columns={'meeting_name': 'recommendation'})
			predictions = predictions.reset_index(drop=True)
			return predictions


class ParamUnavailable(Exception):
	def __init__(self, message):
	  self.message = message


class ColumnsUnavailable(Exception):
	def __init__(self, message):
	  self.message = message