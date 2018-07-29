import datetime

import numpy as np
import pandas as pd

####################
#### TWEAKABLES ####
####################

datafile = 'ml_delai.csv'

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
	processed_features = ml_dataframe[selected_features].copy()

	# Create synthetic features.
	print('Calculating synthetic features...')
	print('1. Calculating dates and times...')
	processed_features['dateheure'] = pd.to_datetime(processed_features['date_oper'].astype(str)+' '+processed_features['heure_oper'].astype(str), format='%Y%m%d %H:%M')
	processed_features['heure_frax'] = processed_features['heure_oper'].str.slice(start=0, stop=2).astype(np.float) + processed_features['heure_oper'].str.slice(start=3, stop=5).astype(np.float)/60
	processed_features['unixdateheure'] = processed_features['dateheure'].astype(np.int64)

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

	# Fillna
	print('4. Preparing people info by filling NaNs...')
	processed_features = processed_features.fillna('')

	print('5. Attributing operations to people...')
	processed_features['ti_shift'] = processed_features.apply(lambda x:x.s8 if x.s9 == '' else x.s9, axis=1)

	processed_features['pharm1'] = processed_features.apply(lambda x: x.s1 if (x.soir==0 and x.fin_de_semaine==0) else (x.s3 if (x.soir==1 and x.fin_de_semaine==0) else (x.s5 if (x.soir==0 and x.fin_de_semaine==1) else (x.s7 if (x.soir==1 and x.fin_de_semaine==1) else('')))), axis=1)

	processed_features['pharm2'] = processed_features.apply(lambda x: x.s2 if (x.soir==0 and x.fin_de_semaine==0) else (x.s4 if (x.soir==1 and x.fin_de_semaine==0) else (x.ti_shift if (x.soir==0 and x.fin_de_semaine==1) else (''))), axis=1)

	processed_features['pharmf'] = processed_features.apply(lambda x: x.f if (x.soir==0 and x.fin_de_semaine==0) else (x.s6 if (x.soir==0 and x.fin_de_semaine==1) else ('')), axis=1)

	processed_features['at_saisie1'] = processed_features.apply(lambda x: x.d1 if (x.soir==0 and x.fin_de_semaine==0) else (x.d7 if (x.soir==1 and x.fin_de_semaine==0) else (x.d11 if (x.soir==0 and x.fin_de_semaine==1) else (x.d18 if (x.soir==1 and x.fin_de_semaine==1) else('')))), axis=1)

	processed_features['at_saisie2'] = processed_features.apply(lambda x: x.d2 if (x.soir==0 and x.fin_de_semaine==0) else (x.d8 if (x.soir==1 and x.fin_de_semaine==0) else (x.d12 if (x.soir==0 and x.fin_de_semaine==1) else (''))), axis=1)

	processed_features['at_fab1'] = processed_features.apply(lambda x: x.f1 if ((x.endroit==1 or x.endroit==6) and x.soir==0 and x.fin_de_semaine==0) else (x.f4 if ((x.endroit==3 or x.endroit==5) and x.soir==0 and x.fin_de_semaine==0) else (x.f7 if (x.endroit>0 and x.soir==1 and x.fin_de_semaine==0) else (x.f11 if ((x.endroit==1 or x.endroit==6) and x.soir==0 and x.fin_de_semaine==1) else (x.f45 if ((x.endroit==3 or x.endroit==5) and x.soir==0 and x.fin_de_semaine==1) else (x.f17 if (x.endroit>0 and x.soir==1 and x.fin_de_semaine==1) else ('')))))), axis=1)

	processed_features['at_fab2'] = processed_features.apply(lambda x: x.f2 if ((x.endroit==1 or x.endroit==6) and x.soir==0 and x.fin_de_semaine==0) else (x.f5 if ((x.endroit==3 or x.endroit==5) and x.soir==0 and x.fin_de_semaine==0) else (x.f12 if ((x.endroit==1 or x.endroit==6) and x.soir==0 and x.fin_de_semaine==1) else (''))), axis=1)

	processed_features=processed_features.drop('ti_shift', axis=1)

	print('6. Stripping strings...')
	processed_features = processed_features.applymap(lambda x:x.strip() if type(x) is str else x)

	return processed_features

def preprocess_targets(ml_dataframe):
	output_targets = pd.DataFrame()
	output_targets[selected_targets] = ml_dataframe[selected_targets]

	return output_targets

####################
##### EXECUTE ######
####################

if __name__ == '__main__':

	print('Reading CSV....')
	ml_delai_dataframe = pd.read_csv(datafile, sep=',')

	print('Processing features...')
	processed_features = preprocess_features(ml_delai_dataframe)
	print('Processing targets...')
	processed_targets = preprocess_targets(ml_delai_dataframe)

	print('Processed features summary:')
	print(processed_features.describe())
	print(processed_features.head())
	print('Saving features...')
	processed_features.to_pickle('features.pkl')

	print('Processed targets summary:')
	print(processed_targets.describe())
	print(processed_targets.head())
	print('Saving targets...')
	processed_targets.to_pickle('targets.pkl')

	correlation_data = pd.concat([processed_features, processed_targets], axis=1, sort=False)
	print('Correlation matrix:')
	print(correlation_data.corr())
