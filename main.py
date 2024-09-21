from utilities import clean_data
from predict import Rec_Model
from rules import *
import os
import datetime
import logging


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
file_handler = logging.FileHandler('main_logs.txt')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# set variables
model_directory = 'model_dir'
model_name = 'model_2021_08_18_13_55'  # newer model creates error due to being retrained with other data
data_directory = 'data'
atx_full_sessions_data = 'ATX_2022-02-15.csv'
atx_history_folder = 'atx_history'
atx_history_fn = 'attended_sessions_history.csv'
predictions_folder = 'predictions'
tam_customers_fn = 'DNA_2022-02-15.csv' # DNA_2021-11-23.csv for DNA file
atx_sessions_with_stages = 'atx_session_stage_key_lookup.csv'
atx_future_schedule = 'atx_upcoming_sessions.csv'


def main():
	path = r'C:\Users\briyue\Documents\atx_recommendations'   ## os.path.abspath(os.getcwd())
	model_path = path + '\\' + model_directory + '\\' + model_name
	df = pd.read_csv(path + '\\' + data_directory + '\\' + atx_full_sessions_data)
	df = clean_data(df)
	r = Rec_Model(model_path, df, 1)
	predictions = r.make_predictions_df()

	df_atxcustinfo = df[['cx_customer_bu_id', 'email_address']]
	df_atxcustinfo = df_atxcustinfo.drop_duplicates(subset=['email_address'])
	df_atxcustinfo = df_atxcustinfo.reset_index(drop=True)

	tam_accounts = pd.read_csv(path + '\\' + data_directory + '\\' + tam_customers_fn)
	tam_accounts.columns = [x.lower() for x in tam_accounts.columns]
	df_tamcustinfo = tam_accounts[['cx_customer_bu_id', 'cx_customer_bu_name', 'offer', 'max_bu_lifecycle_stage', 'days_in_stage', 'cx_theater']]
	df_tamcustinfo = df_tamcustinfo.drop_duplicates(subset=['cx_customer_bu_id', 'max_bu_lifecycle_stage', 'cx_theater'])
	df_tamcustinfo = df_tamcustinfo.rename(columns = {'max_bu_lifecycle_stage' : 'stage'})

	df_joined = predictions.merge(df_atxcustinfo, on='email_address', how='left')
	df_pred_tam = df_joined.merge(df_tamcustinfo, on='cx_customer_bu_id', how='left')

	atxstage_keylookup = pd.read_csv(path + '\\' + data_directory + '\\' + atx_sessions_with_stages)
	atxstage_keylookup = atxstage_keylookup.rename(columns = {'session_name' : 'model_pred_session_name'})
	df_pred_tam = df_pred_tam.rename(columns = {'recommendation' : 'model_pred_session_name'})
	df_pred_0 = df_pred_tam.merge(atxstage_keylookup, on='model_pred_session_name', how='left')
	df_pred_0 = df_pred_0.rename(columns = {'atx_stage' : 'model_pred_session_stage'})

	atx_future_sessions = pd.read_csv(path + '\\' + data_directory + '\\' + atx_future_schedule)
	df_predictions_raw= df_pred_0.merge(atx_future_sessions, on=['offer', 'stage'], how='left')
	rule_predictions_top_stages_agg = df_predictions_raw.groupby(['future_session_name'])['stage'].count().reset_index(name="stage_count")
	rule_predictions_top_stages_agg.to_csv(path + '\\' + predictions_folder + '\\' + 'rule_recommendations_top_stages_aggregated.csv')

	df_predictions_raw = df_predictions_raw.rename(columns = {'future_session_name' : 'rule_recommendation'})
	df_predictions_raw = df_predictions_raw.rename(columns = {'model_pred_session_name' : 'model_recommendation'})
	df_predictions_raw = df_predictions_raw[['cx_theater', 'cx_customer_bu_id', 'cx_customer_bu_name', 'offer', 'stage', 'days_in_stage', 
												'rule_recommendation', 'model_recommendation']]
	df_predictions_raw.to_csv(path + '\\' + predictions_folder + '\\' + 'raw_rule_and_model_recommendations_individual_level.csv')


	prior_attended = pd.read_csv(path + '\\' + atx_history_folder + '\\' + atx_history_fn)
	df_predictions_raw = df_predictions_raw.rename(columns= {'model_recommendation': 'meeting_name'})
	cleanup = Rules(prior_attended, sessionname_col='meeting_name', emailaddress_col='email_address')
	cleaned_predictions = cleanup.remove_prior_attended_sessions(df_predictions=df_predictions_raw)
	cleaned_predictions = cleaned_predictions.rename(columns= {'meeting_name': 'model_recommendation'})
	cleaned_predictions = cleaned_predictions.rename(columns= {'rule_recommendation': 'meeting_name'})
	cleaned_predictions = cleanup.remove_prior_attended_sessions(df_predictions=cleaned_predictions)
	ts = datetime.datetime.today().strftime('%Y_%m_%d_%H_%M')
	filename = 'cleaned_predictions_' + str(ts) + '.csv'
	cleaned_predictions.to_csv(path + '\\' + predictions_folder + '\\' + filename)
	
	
	tam_accounts = tam_accounts.drop_duplicates(subset=['cx_customer_bu_id', 'cx_customer_bu_name'])
	tam_accounts = tam_accounts.reset_index(drop=True)
	static_recommendations = Rules(tam_accounts, stage_col='max_bu_lifecycle_stage', usecase_col='cx_use_case')
	top_stages = static_recommendations.get_top_stages(account_level=False)
	filename = 'rule_preds_top_stages_' + str(tam_customers_fn[:4]) + str(ts) + '.csv'
	top_stages.to_csv(path + '\\' + predictions_folder + '\\' + filename)
	top_use_cases = static_recommendations.get_top_use_cases(account_level=False)
	filename = 'rule_preds_top_use_cases_' + str(tam_customers_fn[:4]) + str(ts) + '.csv'
	top_use_cases.to_csv(path + '\\' + predictions_folder + '\\' + filename)


if __name__ == '__main__':
	main()
