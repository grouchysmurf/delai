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
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder, OneHotEncoder

import preprocess
import scikitplot as skplt

#################
#### CLASSES ####
#################

class data:

	def __init__(self, restrict_data=False):
		# load data
		self.features, self.targets = preprocess.preprocess_features(restrict_data)
		self.targets['delay_cat'] = self.targets['delay'].apply(lambda x: 0 if x < 45 else 1)

	def get_data(self):
		return self.features, self.targets


class keras_model:

	def __init__(self):
		pass

	def define_model(self):

		with tf.device('/cpu:0'):
			model = Sequential()
			model.add(Dense(128, activation='relu'))
			model.add(Dropout(0.5))
			model.add(Dense(128, activation='relu'))
			model.add(Dropout(0.5))
			model.add(Dense(1, activation='sigmoid'))

		if len(K.tensorflow_backend._get_available_gpus()) > 1:
			ondevice_model = multi_gpu_model(model, gpus=len(K.tensorflow_backend._get_available_gpus()))
		else:
			ondevice_model = model

		ondevice_model.compile(loss='binary_crossentropy',
					optimizer='Adam',
					metrics=['binary_accuracy'])

		return ondevice_model


class skl_model:

	def __init__(self, save_timestamp):
		self.skl_save_path = os.path.join(os.getcwd(), 'model')
		pathlib.Path(self.skl_save_path).mkdir(parents=True, exist_ok=True)
		self.keras_save_path = os.path.join(os.getcwd(), 'model')
		pathlib.Path(self.keras_save_path).mkdir(parents=True, exist_ok=True)
		self.save_timestamp = save_timestamp


	def get_split_data(self, features, targets):
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
		pipe = Pipeline([
			('bin', KBinsDiscretizer(n_bins=n)),
		])
		return pipe

	def features_encoding_pipe(self):
		features = []

		string_columns_to_one_hot = ['drug_name', 'user', 'operation_code', 'internal_or_external', 'workbench', 'pharm_disp_1' ,'pharm_disp_2', 'pharm_disp_3', 'pharm_disp_4', 'pharm_disp_5', 'pharm_disp_6', 'pharm_disp_7', 'pharm_disp_8', 'pharm_disp_9', 'pharm_comp', 'tech_disp_1', 'tech_disp_2', 'tech_disp_3', 'tech_disp_4', 'tech_disp_5', 'tech_disp_7', 'tech_disp_8', 'tech_disp_10', 'tech_disp_11', 'tech_disp_12', 'tech_disp_18', 'tech_comp_1', 'tech_comp_2', 'tech_comp_4', 'tech_comp_5', 'tech_comp_6', 'tech_comp_7', 'tech_comp_8', 'tech_comp_11', 'tech_comp_12', 'tech_comp_45', 'tech_comp_17', 'evening', 'weekend_holiday', 'compounding']

		continuous_columns_to_bin = ['time_frax']

		features.append(('string_columns', self.one_hot_pipe(), string_columns_to_one_hot))
		features.append(('continuous_columns_binned', self.n_bin_pipe(16), continuous_columns_to_bin))

		col_transformer_features = ColumnTransformer(transformers=features)
		return col_transformer_features

	def targets_pipe(self):
		pipe = LabelEncoder()
		return pipe
	
	def processing_pipe(self):

		tb_callback = TensorBoard(log_dir=os.path.join(self.keras_save_path, 'tensorboard_logs'))
		checkpoint_callback = ModelCheckpoint(filepath=os.path.join(self.keras_save_path, save_timestamp + 'checkpoints'), verbose=1)
		csv_callback = CSVLogger(filename=os.path.join(self.keras_save_path, save_timestamp + 'history.csv'))
		earlystop_callback = EarlyStopping(monitor='loss', min_delta=0.01, patience=3, verbose=1)

		keras_build_fn = keras_model().define_model

		pipeline = Pipeline([
			('features_encoding', self.features_encoding_pipe()),
			('feature_selection', SelectFromModel(ExtraTreesClassifier(n_estimators=100, n_jobs=-2, verbose=1))),
			('multilayer_perceptron', KerasClassifier(build_fn=keras_build_fn, 
					callbacks=[tb_callback, checkpoint_callback, earlystop_callback, csv_callback])),
		])

		return pipeline


####################
##### EXECUTE ######
####################

if __name__ == '__main__':

	parser = ap.ArgumentParser(description='Try to classify the time required to prepare medications as <45 minutes (short) or > 45 minutes (long)', formatter_class=ap.RawTextHelpFormatter)

	args = parser.parse_args()

	save_timestamp = datetime.now().strftime('%Y%m%d-%H%M')

	# configure environment
	print('Configuring logger...')

	# Logger
	logging_path = os.path.join(os.getcwd(), 'logs', 'build_model')
	pathlib.Path(logging_path).mkdir(parents=True, exist_ok=True)
	logging.basicConfig(
		level=logging.DEBUG,
		format="%(asctime)s [%(levelname)s]  %(message)s",
		handlers=[
			logging.FileHandler(os.path.join(logging_path, save_timestamp + '.log')),
			logging.StreamHandler()
		])
	logging.debug('Logger successfully configured.')

	logging.debug('Obtaining data...')
	features, targets = data(restrict_data=False).get_data()
	s = skl_model(save_timestamp)
	logging.debug('Splitting train and test sets...')
	features_train, features_test, targets_train, targets_test = s.get_split_data(features, targets)
	logging.debug('Preparing targets...')
	le = s.targets_pipe()
	y_train = le.fit_transform(targets_train['delay_cat'])
	y_true = le.transform(targets_test['delay_cat'])
	pipe = s.processing_pipe()
	logging.debug('Fitting model...')
	pipe.fit(features_train, y_train, multilayer_perceptron__batch_size=256, multilayer_perceptron__epochs=100)
	y_pred = pipe.predict(features_test)
	y_probas = pipe.predict_proba(features_test)
	
	logging.debug('Calculating metrics...')

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

	skplt.estimators.plot_feature_importances(pipe.named_steps['feature_selection'].estimator_)
	plt.savefig(os.path.join('model', save_timestamp + 'feature_importance.png'))
