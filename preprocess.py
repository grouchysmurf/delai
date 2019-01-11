import datetime

import numpy as np
import pandas as pd

####################
#### TWEAKABLES ####
####################

datafile = 'data/traintest.csv'

selected_features = [
	 'date_oper'
	,'heure_oper'
	,'soir'
	,'fin_de_semaine'
	,'SORE_USAGER'
	,'SORE_CODE_OPER'
	,'MEDI_NOM'
	,'ORDO_STATUT'
	,'endroit'
	,'s1'
	,'s2'
	,'s3'
	,'s4'
	,'s5'
	,'s6'
	,'s7'
	,'s8'
	,'s9'
	,'f'
	,'d1'
	,'d2'
	,'d3'
	,'d4'
	,'d5'
	,'d7'
	,'d8'
	,'d10'
	,'d11'
	,'d12'
	,'d18'
	,'f1'
	,'f2'
	,'f4'
	,'f5'
	,'f6'
	,'f7'
	,'f8'
	,'f11'
	,'f12'
	,'f45'
	,'f17'
	]

selected_targets = [
	 'delai'
	 ]

pd.options.display.max_rows = 20
pd.options.display.float_format = '{:.1f}'.format

####################
#### FUNCTIONS #####
####################

def preprocess_features(ml_dataframe):  

	# Create synthetic features.
	print('Calculating synthetic features...')
	print('1. Calculating dates and times...')
	ml_dataframe['dateheure'] = pd.to_datetime(ml_dataframe['date_oper'].astype(str)+' '+ml_dataframe['heure_oper'].astype(str), format='%Y%m%d %H:%M')
	ml_dataframe['heure_frax'] = ml_dataframe['heure_oper'].str.slice(start=0, stop=2).astype(np.float) + ml_dataframe['heure_oper'].str.slice(start=3, stop=5).astype(np.float)/60
	ml_dataframe = ml_dataframe.drop(ml_dataframe[ml_dataframe.dateheure < np.datetime64('2018-01-01')].index)
	processed_targets = pd.DataFrame()
	processed_targets[selected_targets] = ml_dataframe[selected_targets]
	processed_features = ml_dataframe.drop('delai', axis=1)

	print('2. Calculating locations...')
	processed_features['fait_a_la_fab'] = processed_features['endroit'].apply(lambda x: 1 if x > 0 else 0)

	print('3. Calculating workload...')
	print('a. Calculating location specific workload... (takes a while...)')
	processed_features['charge_de_travail_endroit'] = processed_features.apply(lambda x: sum(((processed_features['dateheure'] <= x.dateheure) & (processed_features['dateheure'] > (x.dateheure - datetime.timedelta(minutes=90))) & (processed_features['endroit'] == x.endroit))), axis=1)
	print('b. Calculating compounding vs dispensing workload... (takes a while...)')
	processed_features['charge_de_travail_fabounon'] = processed_features.apply(lambda x: sum(((processed_features['dateheure'] <= x.dateheure) & (processed_features['dateheure'] > (x.dateheure - datetime.timedelta(minutes=90))) & (processed_features['endroit'] > 0))) if x.endroit > 0 else sum(((processed_features['dateheure'] <= x.dateheure) & (processed_features['dateheure'] > (x.dateheure - datetime.timedelta(minutes=90))))), axis=1)
	print('c. Calculating order processing workload... (takes a while...)')
	processed_features['charge_de_travail_saisie'] = processed_features.apply(lambda x: sum(((processed_features['dateheure'] <= x.dateheure) & (processed_features['dateheure'] > (x.dateheure - datetime.timedelta(minutes=90))))), axis=1)
	
	# Remove stuff tensorflow can't use (datetimes)
	processed_features = processed_features.drop(['dateheure', 'date_oper', 'heure_oper'], axis=1)

	print('5. Stripping strings...')
	processed_features = processed_features.applymap(lambda x:x.strip() if type(x) is str else x)

	return processed_features, processed_targets

####################
##### EXECUTE ######
####################

if __name__ == '__main__':

	print('Reading CSV....')
	ml_delai_dataframe = pd.read_csv(datafile, sep=',')

	print('Processing features and targets...')
	processed_features, processed_targets = preprocess_features(ml_delai_dataframe)

	print('Processed features summary:')
	print(processed_features.describe())
	print(processed_features.head())
	print('Saving features...')
	processed_features.to_pickle('data/features.pkl')

	print('Processed targets summary:')
	print(processed_targets.describe())
	print(processed_targets.head())
	print('Saving targets...')
	processed_targets.to_pickle('data/targets.pkl')

	correlation_data = pd.concat([processed_features, processed_targets], axis=1, sort=False)
	print('Correlation matrix:')
	print(correlation_data.corr())
