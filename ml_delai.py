import logging
import math
import os
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.utils import multi_gpu_model
from matplotlib import cm, gridspec
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import (KBinsDiscretizer, OneHotEncoder,
                                   OrdinalEncoder, StandardScaler)

##################
#### CLASSES #####
##################

class data:

	def __init__(self, features_file='features.pkl', targets_file='targets.pkl'):
		# load data
		logging.info('Loading data...')
		self.all_features = pd.read_pickle(features_file)
		self.all_targets = pd.read_pickle(targets_file)
		logging.debug('Data loaded.')
		logging.info('Preparing dataframes...')
		self.all_targets['delai_cat'] = self.all_targets['delai'].apply(lambda x: 0 if x < 45 else 1)
		logging.debug('Dataframes ready.')

	def split(self):		
		logging.info('Splitting dataframes into train and test sets...')
		X_train, X_test, y_train, y_test = train_test_split(self.all_features, self.all_targets, train_size=0.8, test_size=0.2)
		logging.debug('Split done.')
		return X_train, X_test, y_train, y_test

class skl_model:

	def __init__(self, X_train, X_test, y_train, y_test):
		self.X_train = X_train
		self.X_test = X_test
		self.y_train = y_train
		self.y_test = y_test
		
	def prepare_data(self):

		transformers_features= []
		
		# encode strings and string-like to one hot vectors
		logging.info('Preparing string processing pipeline...')
		string_preparation_pipe = Pipeline([
			('impute', SimpleImputer(strategy='constant', fill_value='missing')),
			('one_hot_encode', OneHotEncoder(sparse=False, handle_unknown='ignore'))
		])
		logging.info('Preparing string transformer...')
		string_columns_to_transform = ['SORE_USAGER', 'SORE_CODE_OPER', 'MEDI_NOM', 'ORDO_STATUT', 'endroit', 's1' ,'s2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 'f', 'd1', 'd2', 'd3', 'd4', 'd5', 'd7', 'd8', 'd10', 'd11', 'd12', 'd18', 'f1', 'f2', 'f4', 'f5', 'f6', 'f7', 'f8', 'f11', 'f12', 'f45', 'f17']
		transformers_features.append(('string_columns', string_preparation_pipe, string_columns_to_transform))

		# encode some numeric to binned one-hot vectors
		logging.info('Preparing numeric one-hot binning...')
		numeric_bin_pipe = Pipeline([
			('bin', KBinsDiscretizer(n_bins=16)), # open 16 hours a day
		])
		logging.info('Preparing bin transformer...')
		continuous_columns_to_bin = ['unixdateheure', 'heure_frax']
		transformers_features.append(('continuous_columns_binned', numeric_bin_pipe, continuous_columns_to_bin))

		# scale some numeric
		logging.info('Preparing numeric scaling...')
		numeric_scaler = Pipeline([
			('scale', StandardScaler())
		])
		logging.info('Preparine scaler transformer...')
		continuous_columns_to_scale = ['charge_de_travail_endroit', 'charge_de_travail_fabounon', 'charge_de_travail_saisie']
		transformers_features.append(('continuous_columns_scaled', numeric_scaler, continuous_columns_to_scale))

		# encode categorical columns to ordinals
		logging.info('Preparing categorical encoding...')
		categorical_pipe = Pipeline([
			('ordinal', OrdinalEncoder())
		])
		logging.info('Preparing categorical transformer...')
		categorical_columns_to_transform = ['soir', 'fin_de_semaine', 'fait_a_la_fab']
		target_columns_to_transform = ['delai_cat']
		transformers_features.append(('categorical_columns', categorical_pipe, categorical_columns_to_transform))
		# targets are categorical
		transformers_target = [('target_columns', categorical_pipe, target_columns_to_transform)]
		
		# concatenate
		col_transformer_features = ColumnTransformer(transformers=transformers_features)
		col_transformer_targets = ColumnTransformer(transformers=transformers_target)
		logging.info('Transforming features data...')
		transformed_X_train = col_transformer_features.fit_transform(self.X_train)
		transformed_X_test = col_transformer_features.transform(self.X_test)
		logging.info('Done transforming features')
		logging.info('Transforming targets...')
		transformed_y_train = col_transformer_targets.fit_transform(self.y_train)
		transformed_y_test = col_transformer_targets.transform(self.y_test)
		logging.info('Done transforming targets...')
		
		return transformed_X_train, transformed_X_test, transformed_y_train, transformed_y_test

class keras_model:

	def __init__(self, X_train, X_test, y_train, y_test):
		self.X_train = X_train
		self.X_test = X_test
		self.y_train = y_train
		self.y_test = y_test

	def mlp(self):
		logging.info('Building keras model: multilayer perceptron...')
		input_dim = self.X_train.shape[1]
		model = Sequential()
		model.add(Dense(64, input_dim=input_dim, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(64, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(1, activation='sigmoid'))

		model.compile(loss='binary_crossentropy',
					optimizer='rmsprop',
					metrics=['accuracy'])

		logging.info('Training keras model: multilayer perceptron...')
		model.fit(self.X_train, self.y_train,
				epochs=20,
				batch_size=4096)
		logging.info('Scoring keras model: multilayer perceptron...')
		score = model.evaluate(self.X_test, self.y_test, batch_size=4096)
		logging.info('Score of multilayer perceptron on test set is: {}'.format(score[1]))
		model.save('mlp.h5')

	def lstm(self, batch_size, time_steps):
		input_dim = self.X_train.shape[1]
		logging.info('Building keras model: LSTM...')
		model = Sequential()
		model.add(LSTM(128, input_shape=(time_steps, input_dim)))
		model.add(Dropout(0.5))
		model.add(Dense(1, activation='sigmoid'))
		parallel_model = multi_gpu_model(model,gpus=4)

		parallel_model.compile(loss='binary_crossentropy',
					optimizer='Adam',
					metrics=['accuracy'])

		logging.info('Training keras model: LSTM...')
		train_seq = sequence.TimeseriesGenerator(self.X_train, self.y_train, length=time_steps, batch_size=batch_size)
		parallel_model.fit_generator(train_seq, epochs=20)
		logging.info('Scoring keras model: LSTM...')
		test_seq = sequence.TimeseriesGenerator(self.X_test, self.y_test, length=time_steps, batch_size=batch_size)
		score = parallel_model.evaluate_generator(test_seq)
		logging.info('Score of LSTM on test set is: {}'.format(score[1]))
		model.save('lstm.h5')


####################
##### EXECUTE ######
####################

if __name__ == '__main__':

	# configure environment
	print('Configuring logger...')

	# Logger
	logging.basicConfig(
		level=logging.DEBUG,
		format="%(asctime)s [%(levelname)s]  %(message)s",
		handlers=[
			logging.FileHandler('logs/' + datetime.now().strftime('%Y%m%d-%H%M') + '.log'),
			logging.StreamHandler()
		])
	logging.debug('Logger successfully configured.')

	X_train, X_test, y_train, y_test = data().split()
	X_train, X_test, y_train, y_test = skl_model(X_train, X_test, y_train, y_test).prepare_data()
	keras_model(X_train, X_test, y_train, y_test).lstm(5, 32)
