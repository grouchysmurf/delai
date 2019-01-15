# ==============================================================================
# LICENSE GOES HERE
# ==============================================================================

'''
Author: Maxime Thibault

Try to classify the time required to prepare medications as <45 minutes (short) or
> 45 minutes (long)
'''

import argparse as ap
import logging
import os
import pathlib
from datetime import datetime

import numpy as np
import pandas as pd
import scikitplot as skplt
import seaborn as sns
import tensorflow as tf
from joblib import dump, load
from keras import backend as K
from keras.callbacks import (CSVLogger, EarlyStopping, ModelCheckpoint,
                             TensorBoard)
from keras.layers import Dense, Dropout
from keras.models import Sequential, load_model
from keras.utils import multi_gpu_model
from keras.wrappers.scikit_learn import KerasClassifier
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, make_scorer,
                             precision_recall_curve, roc_auc_score)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (KBinsDiscretizer, LabelEncoder,
                                   OneHotEncoder, StandardScaler)

import preprocess

#################
#### CLASSES ####
#################

class data:

	def __init__(self, restrict_data=False):
		# load data
		self.features, self.targets = preprocess.preprocess_features(restrict_data)
		# categorize
		self.targets['delay_cat'] = self.targets['delay'].apply(lambda x: 0 if x < 45 else 1)  # ADJUST CUTOFF HERE

	def get_data(self):
		# return the data
		return self.features, self.targets


class keras_model:

	def __init__(self):
		pass

	def define_model(self):

		# basic multilayer perceptron
		with tf.device('/cpu:0'):
			model = Sequential()
			model.add(Dense(1024, activation='relu'))
			model.add(Dropout(0.2))
			model.add(Dense(32, activation='relu'))
			model.add(Dropout(0.2))
			model.add(Dense(1, activation='sigmoid'))

		# check if gpu
		if len(K.tensorflow_backend._get_available_gpus()) > 1:
			ondevice_model = multi_gpu_model(model, gpus=len(K.tensorflow_backend._get_available_gpus()))
		else:
			ondevice_model = model

		# compile the model
		ondevice_model.compile(loss='binary_crossentropy',
					optimizer='Adam',
					metrics=['binary_accuracy'])

		return ondevice_model


class skl_model:

	def __init__(self):
		# make sure directories required to save everything exist, if not create them
		self.skl_save_path = os.path.join(os.getcwd(), 'model')
		pathlib.Path(self.skl_save_path).mkdir(parents=True, exist_ok=True)
		self.keras_save_path = os.path.join(os.getcwd(), 'model')
		pathlib.Path(self.keras_save_path).mkdir(parents=True, exist_ok=True)

	def get_split_data(self, features, targets):
		# perform a train-test split on the input data
		features_train, features_test, targets_train, targets_test = train_test_split(features, targets)
		return features_train, features_test, targets_train, targets_test
	
	def one_hot_pipe(self):
		# encode strings and string-like to one hot vectors
		pipe = Pipeline([
			('impute', SimpleImputer(strategy='constant', verbose=1)),
			('one_hot_encode', OneHotEncoder(sparse=False, handle_unknown='ignore'))
		])
		return pipe

	def n_bin_pipe(self, n):
		# discretize a continuous variable in a specified number of bins
		pipe = Pipeline([
			('bin', KBinsDiscretizer(n_bins=n)),
		])
		return pipe

	def scaler_pipe(self):
		# scale a continuous variable
		pipe = Pipeline([
			('scaler', StandardScaler())
		])
		return pipe

	def features_encoding_pipe(self):
		# encode features

		features = []

		# adjust columns with data provided by preprocess.py

		string_columns_to_one_hot = ['drug_id', 'user', 'operation_code', 'internal_or_external', 'workbench', 'pharm_disp_1' ,'pharm_disp_2', 'pharm_disp_3', 'pharm_disp_4', 'pharm_disp_5', 'pharm_disp_6', 'pharm_disp_7', 'pharm_disp_8', 'pharm_disp_9', 'pharm_comp', 'tech_disp_1', 'tech_disp_2', 'tech_disp_3', 'tech_disp_4', 'tech_disp_5', 'tech_disp_7', 'tech_disp_8', 'tech_disp_10', 'tech_disp_11', 'tech_disp_12', 'tech_disp_18', 'tech_comp_1', 'tech_comp_2', 'tech_comp_4', 'tech_comp_5', 'tech_comp_6', 'tech_comp_7', 'tech_comp_8', 'tech_comp_11', 'tech_comp_12', 'tech_comp_45', 'tech_comp_17']

		continuous_columns_to_bin = ['time_frax']

		continuous_columns_to_scale = ['workload']

		# build the transformers for the column transformer
		features.append(('string_columns', self.one_hot_pipe(), string_columns_to_one_hot))
		features.append(('continuous_columns_binned', self.n_bin_pipe(16), continuous_columns_to_bin)) # open 16 hours a day, adjust as needed
		features.append(('continuous_columns_scaled', self.scaler_pipe(), continuous_columns_to_scale))

		# build the column transformer
		col_transformer_features = ColumnTransformer(transformers=features)
		return col_transformer_features

	def targets_pipe(self):
		# encode labels
		pipe = LabelEncoder()
		return pipe
	
	def processing_pipe(self, mode):

		# get the keras model build function
		keras_build_fn = keras_model().define_model

		# base steps
		steps = [
			('features_encoding', self.features_encoding_pipe()),
			('feature_selection', SelectFromModel(ExtraTreesClassifier(n_estimators=100, n_jobs=-2, verbose=1))),
		]

		# adjust final step based on desired mode
		if mode == 'ert':
			steps.append(('trees', ExtraTreesClassifier(n_estimators=100, n_jobs=-2, verbose=1))) 
		elif mode =='mlp':
			steps.append(('multilayer_perceptron', KerasClassifier(build_fn=keras_build_fn, batch_size=256, epochs=15))) # mlp in keras to run on gpu using tensorflow if number of samples is high
		
		pipeline = Pipeline(steps)

		return pipeline

class operation_mode:

	def __init__(self, mode):
		self.mode = mode

	def single_train(self, save_timestamp, features, targets):
		# train a model on a traning set and get metrics from the test set
		logging.info('Performing single train...')
		s = skl_model()
		logging.debug('Splitting train and test sets...')
		features_train, features_test, targets_train, targets_test = s.get_split_data(features, targets)
		logging.debug('Preparing targets...')
		le = s.targets_pipe()
		y_train = le.fit_transform(targets_train['delay_cat'])
		y_true = le.transform(targets_test['delay_cat'])
		pipe = s.processing_pipe(self.mode)
		logging.debug('Fitting model...')
		pipe.fit(features_train, y_train)
		y_pred = pipe.predict(features_test)
		y_probas = pipe.predict_proba(features_test)
		
		logging.debug('Calculating metrics...')
		acc = accuracy_score(y_true, y_pred)
		logging.info('Accuracy of model on test set is: {}'.format(acc))
		f1 = f1_score(y_true, y_pred)
		logging.info('F1 Score of model on test set is: {}:'.format(f1))
		auroc = roc_auc_score(y_true, y_probas[:,1])
		logging.info('Area under ROC curve of model on test set is: {}'.format(auroc))
		
		skplt.metrics.plot_confusion_matrix(y_true, y_pred, normalize=True)
		plt.savefig(os.path.join('model', save_timestamp + 'confusion_matrix.png'))
		skplt.metrics.plot_roc(y_true, y_probas)
		plt.savefig(os.path.join('model', save_timestamp + 'roc_curve.png'))
		skplt.metrics.plot_precision_recall(y_true, y_probas)
		plt.savefig(os.path.join('model', save_timestamp + 'precision_recall_curve.png'))

		skplt.estimators.plot_feature_importances(pipe.named_steps['feature_selection'].estimator_, x_tick_rotation=45)
		plt.savefig(os.path.join('model', save_timestamp + 'feature_importance.png'))

	def plot_learning_curve(self, save_timestamp, features, targets):
		# plot the model learning curve on all the data with cross-validation
		logging.info('Performing plotting of learning curve...')

		s = skl_model()
		le = s.targets_pipe()
		y = le.fit_transform(targets['delay_cat'])
		pipe = s.processing_pipe(self.mode)

		skplt.estimators.plot_learning_curve(pipe, features, y, title='Learning Curve', cv=StratifiedKFold(shuffle=True, n_splits=5), scoring='accuracy')
		plt.savefig(os.path.join('model', save_timestamp + 'learning_curve.png'))


####################
##### EXECUTE ######
####################

if __name__ == '__main__':

	# Arguments
	parser = ap.ArgumentParser(description='Try to classify the time required to prepare medications as <45 minutes (short) or > 45 minutes (long)', formatter_class=ap.RawTextHelpFormatter)
	parser.add_argument('--logging_level', metavar='Type_String', type=str, nargs="?", default='info', help='Logging level. Possibilities include "debug" or "info". Metrics are logged with info level, setting level above info will prevent metrics logging.')
	parser.add_argument('--mode', metavar='Type_String', type=str, nargs="?", default='ert', help='Use "mlp" to train a multilayer perceptron binary classifier, "ert" to train an extremly randomized trees classifier.')
	parser.add_argument('--op', metavar='Type_string', type=str, nargs='?', default='st', help='Use "st" to perform a single training pass. Use "lc" to plot a learning curve with cross validation.')
	parser.add_argument('--restrict_data', action='store_true', help='Use this argument to restrict the number of data lines used (for testing.')

	args = parser.parse_args()
	logging_level = args.logging_level
	mode = args.mode
	op = args.op
	restrict_data = args.restrict_data

	# Save timestamp for files generated during a run
	save_timestamp = datetime.now().strftime('%Y%m%d-%H%M')

	# Logger
	print('Configuring logger...')
	if logging_level == 'info':
		ll = logging.INFO
	elif logging_level =='debug':
		ll = logging.DEBUG
	elif logging_level == 'warning':
		ll = logging.WARNING
	logging_path = os.path.join(os.getcwd(), 'logs', 'build_model')
	pathlib.Path(logging_path).mkdir(parents=True, exist_ok=True)
	logging.basicConfig(
		level=ll,
		format="%(asctime)s [%(levelname)s]  %(message)s",
		handlers=[
			logging.FileHandler(os.path.join(logging_path, save_timestamp + '.log')),
			logging.StreamHandler()
		])
	logging.debug('Logger successfully configured.')
	if logging_level == 'warning':
		logging.warning('Logging level is set at WARNING. Metrics are logged at level INFO. Metrics will not be logged.')

	logging.info('Using mode: {}'.format(mode))
	
	# Get the data
	logging.debug('Obtaining data...')
	features, targets = data(restrict_data=restrict_data).get_data()
	logging.debug('Obtained {} samples for features and {} samples for targets'.format(len(features), len(targets)))

	# Execute
	o = operation_mode(mode)
	if op == 'st':
		o.single_train(save_timestamp, features, targets)
	elif op == 'lc':
		o.plot_learning_curve(save_timestamp, features, targets)

	quit()
