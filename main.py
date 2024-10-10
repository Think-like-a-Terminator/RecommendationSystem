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
model_name = 'model_2021_08_18_13_55'  # replace model_name with the trained model from train_main in model.py
data_directory = 'data'
data_file = 'data_for_recommenders.csv'
predictions_folder = 'predictions'


# run main after running train_main function from model.py
def main():
	path = os.path.abspath(os.getcwd())
	model_path = path + '\\' + model_directory + '\\' + model_name
	df = pd.read_csv(path + '\\' + data_directory + '\\' + data_file)
	df = clean_data(df)
	r = Rec_Model(model_path, df, 1)
	predictions = r.make_predictions_df()
	predictions.to_csv(path + '\\' + predictions_folder + '\\' + 'recommendations.csv')

if __name__ == '__main__':
	main()
