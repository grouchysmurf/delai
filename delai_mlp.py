# ==============================================================================
# LICENSE GOES HERE
# ==============================================================================

'''
Author: Maxime Thibault

Try to predict the time between operation and dispensation for hospital pharmacy
drugs
'''

import logging
import os
import pathlib
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from joblib import dump, load
from keras import backend as K
from keras.callbacks import (CSVLogger, EarlyStopping, ModelCheckpoint,
                             TensorBoard)
from keras.layers import Dense, Dropout, Input, LSTM
from keras.models import Model, load_model
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.utils import Sequence, multi_gpu_model
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (KBinsDiscretizer, OneHotEncoder,
                                   OrdinalEncoder, StandardScaler)

#################
#### CLASSES ####
#################

class data:

	def __init__(self, features_file='data/features_old.pkl', targets_file='data/targets_old.pkl'):
		# load data
		logging.info('Loading data...')
		self.all_features = pd.read_pickle(features_file)
		self.all_targets = pd.read_pickle(targets_file)
		logging.debug('Data loaded.')
		logging.info('Preparing dataframes...')
		self.all_targets['delai_cat'] = self.all_targets['delai'].apply(lambda x: 0 if x < 45 else 1)
		logging.debug('Dataframes ready.')

	def get_data(self):
		logging.info('Returning data...')
		return self.all_features, self.all_targets


class skl_model:

	def __init__(self):
		self.skl_save_path = os.path.join(os.getcwd(), 'model')
		pathlib.Path(self.skl_save_path).mkdir(parents=True, exist_ok=True)

	def one_hot_pipe(self):
		# encode strings and string-like to one hot vectors
		logging.info('Preparing imputing missing and one-hot encoding pipeline...')
		pipe = Pipeline([
			('impute', SimpleImputer(strategy='constant')),
			('one_hot_encode', OneHotEncoder(sparse=False, handle_unknown='ignore'))
		])
		return pipe

	def n_bin_pipe(self, n):
		logging.info('Preparing one-hot binning for {} bins pipeline...'.format(n))
		pipe = Pipeline([
			('bin', KBinsDiscretizer(n_bins=n)),
		])
		return pipe

	def straight_input_pipe(self):
		straight_input_features = []

		logging.info('Preparing string transformer...')
		string_columns_to_transform = ['MEDI_NOM', 'SORE_USAGER', 'SORE_CODE_OPER', 'ORDO_STATUT', 'endroit', 's1' ,'s2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 'f', 'd1', 'd2', 'd3', 'd4', 'd5', 'd7', 'd8', 'd10', 'd11', 'd12', 'd18', 'f1', 'f2', 'f4', 'f5', 'f6', 'f7', 'f8', 'f11', 'f12', 'f45', 'f17']
		straight_input_features.append(('string_columns', self.one_hot_pipe(), string_columns_to_transform))
		
		logging.info('Preparing bin transformer...')
		continuous_columns_to_bin = ['heure_frax']
		straight_input_features.append(('continuous_columns_binned', self.n_bin_pipe(16), continuous_columns_to_bin))

		logging.info('Preparing categorical transformer...')
		categorical_columns_to_transform = ['soir', 'fin_de_semaine', 'fait_a_la_fab']
		straight_input_features.append(('categorical_columns', self.one_hot_pipe(), categorical_columns_to_transform))
		
		col_transformer_features = ColumnTransformer(transformers=straight_input_features)
		return col_transformer_features

	def targets_pipe(self):
		logging.info('Preparing target encoding pipe...')
		target_columns_to_transform = ['delai_cat']
		transformers_target = [('target_columns', OrdinalEncoder(), target_columns_to_transform)]
		col_transformer_targets = ColumnTransformer(transformers=transformers_target)
		return col_transformer_targets
	
	def prepare_data(self, features, targets, valid=False):

		if valid == False:
			try:
				logging.debug('Trying to obtain previous random state from: {}'.format(self.skl_save_path))
				random_state = load(os.path.join(self.skl_save_path, 'random_state.joblib'))
				logging.info('Successfully loaded previous random state.')
			except Exception as e:
				logging.warning('Loading previous random state failed because of exception: {}. Generating new random state and saving.'.format(e))
				random_state = np.random.randint(100)
				dump(random_state, os.path.join(self.skl_save_path, 'random_state.joblib'))

			logging.info('Splitting data into train and test sets...')
			features_train, features_test, targets_train, targets_test = train_test_split(features, targets, random_state=random_state, train_size=0.9, test_size=0.1)

		try:
			logging.debug('Trying to obtain previously fitted straight input pipe from: {}'.format(self.skl_save_path))
			encoding_pipe = load(os.path.join(self.skl_save_path, 'encoding_pipe.joblib'))
			if valid == False:
				logging.info('Successfully loaded previously fitted encoding pipe. Transforming input encoding train data...')
				X_encoding_train = encoding_pipe.transform(features_train)
			else:
				logging.info('Successfully loaded previously fitted encoding pipe. Transforming input VALIDATION data...')
				X_valid = encoding_pipe.transform(features)
		except Exception as e:
			if valid == False:
				logging.warning('Loading previously fitted straight input pipe failed because of exception: {}. Fitting and saving new pipe.'.format(e))
				encoding_pipe = self.straight_input_pipe()
				X_encoding_train = encoding_pipe.fit_transform(features_train)
				dump(encoding_pipe, os.path.join(self.skl_save_path, 'encoding_pipe.joblib'))
			else:
				logging.critical('Loading previously fitted straight input pipe failed because of exception: {}. Cannot transform validation data. Quitting...'.format(e))
				quit()
		if valid == False:
			logging.info('Transforming input encoding test data...')
			X_encoding_test = encoding_pipe.transform(features_test)
		'''
		try:
			logging.debug('Trying to obtain previously fitted label encoder from: {}'.format(self.skl_save_path))
			targets_pipe = load(os.path.join(self.skl_save_path, 'targets_pipe.joblib'))
			if valid == False:
				logging.info('Successfully loaded previously fitted label encoder. Transforming target data...')
				y_train = targets_pipe.transform(targets_train)
			else:
				logging.info('Successfully loaded previously fitted label encoder. Transforming target VALIDATION data...')
				y_valid = targets_pipe.transform(targets)
		except Exception as e:
			if valid == False:
				logging.warning('Loading previously fitted label encoder pipe failed because of exception: {}. Fitting and saving new pipe.'.format(e))
				targets_pipe = self.targets_pipe()
				y_train = targets_pipe.fit_transform(targets_train)
				dump(targets_pipe, os.path.join(self.skl_save_path, 'targets_pipe.joblib'))
			else:
				logging.critical('Loading previously fitted label encoder pipe failed because of exception: {}. Cannot transform validation data. Quitting...'.format(e))
				quit()'''
		if valid == False:
			#logging.info('Transforming input label encoder test data...')
			y_train = targets_train['delai'].values
			y_test = targets_test['delai'].values #targets_pipe.transform(targets_test)
		else:
			y_valid = targets['delai'].values

		if valid == False:		
			return X_encoding_train, X_encoding_test, y_train, y_test, features_test, targets_test
		else:
			return X_valid, y_valid

class keras_model:

	def __init__(self):
		self.keras_save_path = os.path.join(os.getcwd(), 'model')
		pathlib.Path(self.keras_save_path).mkdir(parents=True, exist_ok=True)

	def define_model(self, straight_input_dim, batch_size, sequence_length):
		logging.debug('Building model...')

		straight_input = Input(batch_shape=(batch_size, sequence_length, straight_input_dim), dtype='float32', name='straight_input')
		stack = LSTM(128, return_sequences=True, stateful=True)(straight_input)
		stack = LSTM(128, return_sequences=True, stateful=True)(stack)
		stack = LSTM(128, stateful=True)(stack)
		stack = Dropout(0.5)(stack)
		stack = Dense(128, activation='relu')(stack)
		stack = Dropout(0.5)(stack)
		output = Dense(1, activation='linear', name='main_output')(stack)

		with tf.device('/cpu:0'):
			self.model = Model(inputs=[straight_input], outputs=[output])

		if len(K.tensorflow_backend._get_available_gpus()) > 1:
			self.ondevice_model = multi_gpu_model(self.model, gpus=len(K.tensorflow_backend._get_available_gpus()))
		else:
			self.ondevice_model = self.model

		self.ondevice_model.compile(loss='mean_squared_error',
					optimizer='Adam',
					metrics=['mean_absolute_error'])

	def load_or_train_model(self, X_encoding_train=None, X_encoding_test=None, y_train=None, y_test=None, evaluate=False, batch_size=32, sequence_length=10, epochs=None):
		data_gen_train = TimeseriesGenerator(X_encoding_train, y_train, length=sequence_length, batch_size=batch_size)
		data_gen_test = TimeseriesGenerator(X_encoding_test, y_test, length=sequence_length, batch_size=batch_size)
		try:
			logging.debug('Trying to obtain previously fitted Keras MultiLayer Perceptron from: {}'.format(self.keras_save_path))
			self.ondevice_model = load_model(os.path.join(self.keras_save_path, 'model.h5'))
			logging.info('Successfully loaded previously fitted MultiLayer Perceptron.')
		except Exception as e:
			logging.warning('Loading previously fitted MultiLayer Perceptron failed because of exception: {}. Training new model...'.format(e))
			logging.info('Training keras model...')
			tb_callback = TensorBoard(log_dir=os.path.join(self.keras_save_path, 'tensorboard_logs'))
			checkpoint_callback = ModelCheckpoint(filepath=os.path.join(self.keras_save_path, datetime.now().strftime('%Y%m%d-%H%M') + 'checkpoints'), verbose=1)
			csv_callback = CSVLogger(filename=os.path.join(self.keras_save_path, datetime.now().strftime('%Y%m%d-%H%M') + 'history.csv'))
			earlystop_callback = EarlyStopping(monitor='val_loss', min_delta=4, patience=3, verbose=1)
			self.ondevice_model.fit_generator(data_gen_train,
					epochs=epochs,
					callbacks=[tb_callback, checkpoint_callback, earlystop_callback, csv_callback],
					validation_data=data_gen_test)
			self.ondevice_model.save(os.path.join(self.keras_save_path, 'model.h5'))
			logging.info('Scoring keras model...')

		if evaluate == True:
			score = self.ondevice_model.evaluate_generator(data_gen_test)
			#auroc = roc_auc_score(y_test, y_pred)
			logging.info('Loss of model on test set is: {}'.format(score[0]))
			logging.info('Metric of model on test set is: {}'.format(score[1]))
			#logging.info('Area under ROC curve of model on test set is: {}'.format(auroc))
			y_pred = self.ondevice_model.predict_generator(data_gen_test)
			y_pred_flat = y_pred.ravel()
			d = {'test_y_true':y_test, 'test_y_pred':y_pred_flat}
			test_results_df = pd.DataFrame(data=d, columns=['test_y_true', 'test_y_pred'])
		else:
			test_results_df=None

		return test_results_df


####################
##### EXECUTE ######
####################

if __name__ == '__main__':

	# configure environment
	print('Configuring logger...')

	# Logger
	logging_path = os.path.join(os.getcwd(), 'logs', 'build_model')
	pathlib.Path(logging_path).mkdir(parents=True, exist_ok=True)
	logging.basicConfig(
		level=logging.DEBUG,
		format="%(asctime)s [%(levelname)s]  %(message)s",
		handlers=[
			logging.FileHandler(os.path.join(logging_path, datetime.now().strftime('%Y%m%d-%H%M') + '.log')),
			logging.StreamHandler()
		])
	logging.debug('Logger successfully configured.')
	
	batch_size = 256
	sequence_length=10
	epochs = 100

	features, targets = data().get_data()
	valid_features, valid_targets = data(features_file='data/features_valid.pkl', targets_file='data/targets_valid.pkl').get_data()
	X_encoding_train, X_encoding_test, y_train, y_test, features_test, targets_test = skl_model().prepare_data(features, targets)
	X_valid, y_valid = skl_model().prepare_data(valid_features, valid_targets, valid=True)

	straight_input_dim = len(X_encoding_train[0])

	m = keras_model()
	m.define_model(straight_input_dim, batch_size, sequence_length)
	test_results_df = m.load_or_train_model(X_encoding_train=X_encoding_train, X_encoding_test=X_encoding_test, y_train=y_train, y_test=y_test, batch_size=batch_size, sequence_length=sequence_length, epochs=epochs, evaluate=True)
	
	sns.set(style='darkgrid')
	sns.set_palette('muted')
	fig = sns.jointplot(x=test_results_df['test_y_true'], y=test_results_df['test_y_pred'], xlim=(0,100), ylim=(0,100), kind='kde')
	#fig = sns.FacetGrid(row='SORE_CODE_OPER', col='fait_a_la_fab', xlim=(0,100), ylim=(0,100))
	#fig.map(kdeplot, test_results_df['test_y_true'], test_results_df['test_y_pred'])
	plt.show()
	
	quit()
