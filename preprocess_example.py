# ==============================================================================
# LICENSE GOES HERE
# ==============================================================================

'''
Author: Maxime Thibault

Preprocess data extracted from pharmacy databases as csv files to pandas dataframes
for input into the machine learning algorithm.
'''

import datetime

import numpy as np
import pandas as pd

####################
#### FUNCTIONS #####
####################

def preprocess_features(datafile, restrict_data):  
	
	selected_targets = [
		'delai'
		]
	
	print('Reading CSV....')
	ml_dataframe = pd.read_csv(datafile, sep=',', index_col=False) # Tune CSV reading parameters here
	if restrict_data == True: # Use this flag during debugging to reduce processing time. Results will be poor if this flag is enabled
		print('Data restriction enabled !')
		ml_dataframe = ml_dataframe[-50000:] # This number can be tuned according to the amount of data available

	# Create synthetic features.
	print('Calculating synthetic features...')
	print('1. Calculating dates and times...')
	# Convert date (string like 2004-03-03) and time (string like 06:28) columns to a pandas datetime index
	ml_dataframe['datetime'] = pd.to_datetime(ml_dataframe['DATE_COLUMN'].astype(str)+' '+ml_dataframe['TIME_COLUMN'].astype(str), format='%Y%m%d %H:%M')
	ml_dataframe.sort_values(by=['datetime'], inplace=True)
	ml_dataframe.set_index('datetime', inplace=True)
	ml_dataframe['time_frax'] = ml_dataframe['TIME_COLUMN'].str.slice(start=0, stop=2).astype(np.float) + ml_dataframe['TIME_COLUMN'].str.slice(start=3, stop=5).astype(np.float)/60

	print('2. Separating targets')
	# Take the targets and send them to their own dataframe
	processed_targets = pd.DataFrame()
	processed_targets[selected_targets] = ml_dataframe[selected_targets]
	processed_features = ml_dataframe.drop(selected_targets, axis=1)

	print('3. Calculating workload...')
	# Perform the folling count for workload
	processed_features['workload'] = processed_features['OPERATION_ID_COLUMN'].rolling(window='2700s').count()

	print('4. Renaming variables...')
	# Rename feature columns from the output of the pharmacy databases to the expected inputs of the machine learning algorithm
	features_rename_dict = {'DRUG_NAME_COLUMN':'drug_id', 'USER_NAME_COLUMN':'user', 'OPERATION_ID_COLUMN':'operation_code', 'PATIENT_STATUS_COLUMN':'internal_or_external', 'WORKBENCH_ID_COLUMN':'workbench', 'PHARMACIST_DISPENSATION1_COLUMN':'pharm_disp_1' ,'PHARMACIST_DISPENSATION2_COLUMN':'pharm_disp_2', 'TECH_DISPENSATION1_COLUMN':'tech_disp_1', 'TECH_DISPENSATION2_COLUMN':'tech_disp_2'}
	targets_rename_dict = {'DELAY_COLUMN': 'delay'}

	processed_features = processed_features.rename(index=str, columns=features_rename_dict)
	processed_targets = processed_targets.rename(index=str, columns=targets_rename_dict)

	return processed_features, processed_targets
