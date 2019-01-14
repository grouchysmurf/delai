import datetime

import numpy as np
import pandas as pd

####################
#### FUNCTIONS #####
####################

def preprocess_features(restrict_data):  
	
	datafile = 'data/traintest.csv'

	selected_targets = [
		'delai'
		]
	
	print('Reading CSV....')
	ml_dataframe = pd.read_csv(datafile, sep=',', index_col=False)
	if restrict_data == True:
		ml_dataframe = ml_dataframe[:10000]

	# Create synthetic features.
	print('Calculating synthetic features...')
	print('1. Calculating dates and times...')
	ml_dataframe['datetime'] = pd.to_datetime(ml_dataframe['date_oper'].astype(str)+' '+ml_dataframe['heure_oper'].astype(str), format='%Y%m%d %H:%M')
	ml_dataframe.sort_values(by=['datetime'], inplace=True)
	ml_dataframe.set_index('datetime', inplace=True)
	ml_dataframe['time_frax'] = ml_dataframe['heure_oper'].str.slice(start=0, stop=2).astype(np.float) + ml_dataframe['heure_oper'].str.slice(start=3, stop=5).astype(np.float)/60
	processed_targets = pd.DataFrame()
	processed_targets[selected_targets] = ml_dataframe[selected_targets]
	processed_features = ml_dataframe.drop(selected_targets, axis=1)

	print('2. Calculating locations...')
	processed_features['compounding'] = processed_features['endroit'].apply(lambda x: 1 if x > 0 else 0)

	print('3. Calculating workload...')
	processed_features['workload'] = processed_features['SORE_NOAUTO'].rolling(window='3600s').count()

	# Remove stuff tensorflow can't use (datetimes)
	processed_features = processed_features.drop(['date_oper', 'heure_oper'], axis=1)

	print('4. Stripping strings...')
	processed_features = processed_features.applymap(lambda x:x.strip() if type(x) is str else x)

	print('5. Renaming variables...')
	features_rename_dict = {'MEDI_NOM':'drug_name', 'SORE_USAGER':'user', 'SORE_CODE_OPER':'operation_code', 'ORDO_STATUT':'internal_or_external', 'endroit':'workbench', 's1':'pharm_disp_1' ,'s2':'pharm_disp_2', 's3':'pharm_disp_3', 's4':'pharm_disp_4', 's5':'pharm_disp_5', 's6':'pharm_disp_6', 's7':'pharm_disp_7', 's8':'pharm_disp_8', 's9':'pharm_disp_9', 'f':'pharm_comp', 'd1':'tech_disp_1', 'd2':'tech_disp_2', 'd3':'tech_disp_3', 'd4':'tech_disp_4', 'd5':'tech_disp_5', 'd7':'tech_disp_7', 'd8':'tech_disp_8', 'd10':'tech_disp_10', 'd11':'tech_disp_11', 'd12':'tech_disp_12', 'd18':'tech_disp_18', 'f1':'tech_comp_1', 'f2':'tech_comp_2', 'f4':'tech_comp_4', 'f5':'tech_comp_5', 'f6':'tech_comp_6', 'f7':'tech_comp_7', 'f8':'tech_comp_8', 'f11':'tech_comp_11', 'f12':'tech_comp_12', 'f45':'tech_comp_45', 'f17':'tech_comp_17', 'soir':'evening', 'fin_de_semaine':'weekend_holiday', 'fait_a_la_fab':'compounding'}
	targets_rename_dict = {'delai': 'delay'}

	processed_features = processed_features.rename(index=str, columns=features_rename_dict)
	processed_targets = processed_targets.rename(index=str, columns=targets_rename_dict)

	return processed_features, processed_targets
