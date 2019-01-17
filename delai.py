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
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.models import Sequential, load_model
from keras.regularizers import l1_l2
from keras.utils import multi_gpu_model
from keras.wrappers.scikit_learn import KerasClassifier
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (accuracy_score, f1_score, make_scorer,
							 precision_recall_curve, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
									 StratifiedKFold, TimeSeriesSplit,
									 cross_validate)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (KBinsDiscretizer, LabelEncoder,
								   OneHotEncoder, StandardScaler)

import preprocess

#################
#### CLASSES ####
#################

class data:

	def __init__(self, datafile, restrict_data=False):
		# load data
		self.features, self.targets = preprocess.preprocess_features(datafile, restrict_data)
		# categorize
		self.targets['delay_cat'] = self.targets['delay'].apply(lambda x: 0 if x < 45 else 1)  # ADJUST CUTOFF HERE

	def get_data(self):
		# return the data
		return self.features, self.targets


class keras_model:

	def __init__(self):
		pass

	def define_model(self, layer_1_size=64, layer_2_size=64, dropout_rate=0.5, l1_reg=0, l2_reg=0.01):

		# basic multilayer perceptron
		with tf.device('/cpu:0'):
			model = Sequential()
			model.add(Dense(layer_1_size, activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))
			model.add(Dropout(dropout_rate))
			model.add(Dense(layer_2_size, activation='relu', kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))
			model.add(Dropout(dropout_rate))
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

		# base steps
		steps = [
			('features_encoding', self.features_encoding_pipe()),
			('feature_selection', RFE(RandomForestClassifier(n_estimators=10, n_jobs=-2, verbose=1), step=0.2, verbose=1)),
		]

		# adjust final step based on desired mode
		if mode == 'ert':
			steps.append(('clf', ExtraTreesClassifier(n_estimators=10, n_jobs=-2, verbose=1))) 
		elif mode == 'rf':
			steps.append(('clf', RandomForestClassifier(n_estimators=10, n_jobs=-2, verbose=1)))
		elif mode =='mlp':
			steps.append(('clf', KerasClassifier(build_fn=keras_model().define_model, batch_size=256, epochs=20)))
		elif mode == 'sgd':
			steps.append(('clf', SGDClassifier(loss='log', max_iter=1000, tol=0.005, shuffle=True, verbose=1, n_jobs=-2, early_stopping=True, n_iter_no_change=3)))
				
		pipeline = Pipeline(steps)

		return pipeline

class operation_mode:

	def __init__(self, mode):
		self.mode = mode

	def single_train_save(self, save_timestamp, features, targets):
		logging.info('Performing single train...')
		s = skl_model()
		logging.debug('Preparing targets...')
		le = s.targets_pipe()
		y = le.fit_transform(targets['delay_cat'])
		pipe = s.processing_pipe(self.mode)
		logging.debug('Fitting model...')
		pipe.fit(features, y)

		logging.info('Saving...')
		dump([pipe, le], os.path.join('model', save_timestamp + 'model.joblib'))

	def single_cross_validation(self, save_timestamp, features, targets):
		logging.info('Performing single cross-validation')
		s = skl_model()
		logging.debug('Preparing targets...')
		le = s.targets_pipe()
		y = le.fit_transform(targets['delay_cat'])
		pipe = s.processing_pipe(self.mode)
		logging.debug('Cross-validating...')
		cv_results_dict = cross_validate(pipe, features, y, scoring=['accuracy', 'f1', 'roc_auc'], cv=TimeSeriesSplit(n_splits=5), verbose=1)
		cv_results_df = pd.DataFrame.from_dict(cv_results_dict)
		cv_results_filtered = cv_results_df[['train_f1', 'train_roc_auc', 'train_accuracy', 'test_f1', 'test_roc_auc', 'test_accuracy']].copy()
		cv_results_filtered.rename(inplace=True, index=str, columns={'train_f1': 'Train F1', 'train_roc_auc':'Train AUROC', 'train_accuracy':'Train accuracy', 'test_f1': 'Test F1', 'test_roc_auc':'Test AUCROC', 'test_accuracy':'Test accuracy'})
		cv_results_graph_df = cv_results_filtered.stack().reset_index()
		cv_results_graph_df.rename(inplace=True, index=str, columns={'level_0':'Split', 'level_1':'Metric', 0:'Result'})
		sns.relplot(x='Split', y='Result', hue='Metric', kind='line', data=cv_results_graph_df)
		plt.savefig(os.path.join('model', save_timestamp + 'crossval_results.png'))

	def random_cross_validation(self, save_timestamp, features, targets, n_iter):
		logging.info('Performing random search with cross validation')
		s = skl_model()
		logging.debug('Preparing targets...')
		le = s.targets_pipe()
		y = le.fit_transform(targets['delay_cat'])
		pipe = s.processing_pipe(self.mode)
		logging.debug('Starting search...')
		grid = dict(feature_selection=[None, 
							RFE(RandomForestClassifier(n_estimators=10, n_jobs=-2, verbose=1, max_depth=10), step=0.1, verbose=1),
							RFE(RandomForestClassifier(n_estimators=100, n_jobs=-2, verbose=1, max_depth=10), step=0.1, verbose=1),
							RFE(RandomForestClassifier(n_estimators=10, n_jobs=-2, verbose=1, max_depth=100), step=0.1, verbose=1),
							RFE(RandomForestClassifier(n_estimators=100, n_jobs=-2, verbose=1, max_depth=100), step=0.1, verbose=1),
							SelectFromModel(RandomForestClassifier(n_estimators=10, n_jobs=-2, verbose=1, max_depth=10)),
							SelectFromModel(RandomForestClassifier(n_estimators=100, n_jobs=-2, verbose=1, max_depth=10)),
							SelectFromModel(RandomForestClassifier(n_estimators=10, n_jobs=-2, verbose=1, max_depth=100)),
							SelectFromModel(RandomForestClassifier(n_estimators=100, n_jobs=-2, verbose=1, max_depth=100))],
					clf=[RandomForestClassifier(n_estimators=10, n_jobs=-2, verbose=1, max_depth=10),
							RandomForestClassifier(n_estimators=100, n_jobs=-2, verbose=1, max_depth=10),
							RandomForestClassifier(n_estimators=10, n_jobs=-2, verbose=1, max_depth=100),
							RandomForestClassifier(n_estimators=100, n_jobs=-2, verbose=1, max_depth=100),
							KerasClassifier(build_fn=keras_model().define_model, batch_size=256, epochs=1000, layer_1_size=64, layer_2_size=64, dropout_rate=0.5, l1_reg=0, l2_reg=0.01, callbacks=[EarlyStopping(monitor='loss', min_delta=0.005, patience=3, verbose=1)]),
							KerasClassifier(build_fn=keras_model().define_model, batch_size=1024, epochs=1000, layer_1_size=64, layer_2_size=64, dropout_rate=0.5, l1_reg=0, l2_reg=0.01, callbacks=[EarlyStopping(monitor='loss', min_delta=0.005, patience=3, verbose=1)]),
							KerasClassifier(build_fn=keras_model().define_model, batch_size=256, epochs=1000, layer_1_size=128, layer_2_size=128, dropout_rate=0.5, l1_reg=0, l2_reg=0.01, callbacks=[EarlyStopping(monitor='loss', min_delta=0.005, patience=3, verbose=1)]),
							KerasClassifier(build_fn=keras_model().define_model, batch_size=1024, epochs=1000, layer_1_size=128, layer_2_size=128, dropout_rate=0.5, l1_reg=0, l2_reg=0.01, callbacks=[EarlyStopping(monitor='loss', min_delta=0.005, patience=3, verbose=1)]),
							KerasClassifier(build_fn=keras_model().define_model, batch_size=256, epochs=1000, layer_1_size=128, layer_2_size=128, dropout_rate=0.5, l1_reg=0, l2_reg=0.05, callbacks=[EarlyStopping(monitor='loss', min_delta=0.005, patience=3, verbose=1)]),
							KerasClassifier(build_fn=keras_model().define_model, batch_size=1024, epochs=1000, layer_1_size=128, layer_2_size=128, dropout_rate=0.5, l1_reg=0, l2_reg=0.05, callbacks=[EarlyStopping(monitor='loss', min_delta=0.005, patience=3, verbose=1)]),
							SGDClassifier(loss='log', max_iter=1000, tol=0.005, shuffle=True, verbose=1, n_jobs=-2, early_stopping=True, n_iter_no_change=3)])
		rcv = RandomizedSearchCV(pipe, grid, n_iter=n_iter, scoring=['accuracy', 'f1', 'roc_auc'], cv=TimeSeriesSplit(n_splits=5), verbose=1, error_score=0, return_train_score=True, refit='accuracy')
		rcv.fit(features, y)
		rcv_results_df = pd.DataFrame.from_dict(rcv.cv_results_)
		rcv_results_df.to_csv(os.path.join('model', save_timestamp + 'random_cross_val_results.csv'))
		rcv_results_filtered = rcv_results_df[['split0_test_accuracy', 'split1_test_accuracy', 'split2_test_accuracy','split3_test_accuracy', 'split4_test_accuracy', 'split0_test_f1', 'split1_test_f1', 'split2_test_f1', 'split3_test_f1', 'split4_test_f1', 'split0_test_roc_auc', 'split1_test_roc_auc', 'split2_test_roc_auc', 'split3_test_roc_auc','split4_test_roc_auc']].copy()
		rcv_results_filtered.rename(inplace=True, index=str, columns={'split0_test_accuracy':'Test accuracy', 'split1_test_accuracy':'Test accuracy', 'split2_test_accuracy':'Test accuracy', 'split3_test_accuracy':'Test accuracy', 'split4_test_accuracy':'Test accuracy', 'split0_test_f1':'Test F1', 'split1_test_f1':'Test F1', 'split2_test_f1':'Test F1', 'split3_test_f1':'Test F1', 'split4_test_f1':'Test F1', 'split0_test_roc_auc':'Test AUROC', 'split1_test_roc_auc':'Test AUROC', 'split2_test_roc_auc':'Test AUROC', 'split3_test_roc_auc':'Test AUROC','split4_test_roc_auc':'Test AUROC'})
		rcv_results_graph_df = rcv_results_filtered.stack().reset_index()
		rcv_results_graph_df.rename(inplace=True, index=str, columns={'level_0':'Iteration', 'level_1':'Metric', 0:'Result'})
		sns.relplot(x='Iteration', y='Result', hue='Metric', kind='line', ci='sd', data=rcv_results_graph_df)
		plt.savefig(os.path.join('model', save_timestamp + 'random_grid_search_results.png'))
		
	def valid(self, save_timestamp, model_file, features, targets):
		logging.debug('Loading model from file: {}...'.format(model_file))
		try:
			pipe, le = load(model_file)
			logging.debug('Model successfully loaded from file.')
		except Exception as e:
			logging.critical('Model could not be loaded because of exception: {} . Quitting...'.format(e))
			quit()
		y_true = le.transform(targets['delay_cat'])
		y_pred = pipe.predict(features)
		y_probas = pipe.predict_proba(features)
		self.plot_metrics(save_timestamp, y_true, y_pred, y_probas)

	def plot_metrics(self, save_timestamp, y_true, y_pred, y_probas):
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

		# skplt.estimators.plot_feature_importances(pipe.named_steps['feature_selection'].estimator_, x_tick_rotation=45)
		# plt.savefig(os.path.join('model', save_timestamp + 'feature_importance.png'))
	
	def plot_learning_curve(self, save_timestamp, features, targets):
		# plot the model learning curve on all the data with cross-validation
		logging.info('Performing plotting of learning curve...')

		s = skl_model()
		le = s.targets_pipe()
		y = le.fit_transform(targets['delay_cat'])
		pipe = s.processing_pipe(self.mode)

		skplt.estimators.plot_learning_curve(pipe, features, y, title='Learning Curve', cv=TimeSeriesSplit(n_splits=5), scoring='accuracy')
		plt.savefig(os.path.join('model', save_timestamp + 'learning_curve.png'))


####################
##### EXECUTE ######
####################

if __name__ == '__main__':

	parser = ap.ArgumentParser(description='Try to classify the time required to prepare medications as <45 minutes (short) or > 45 minutes (long)', formatter_class=ap.RawTextHelpFormatter)
	parser.add_argument('--verbose', action='store_true', help='Use this argument to log at DEBUG level, otherwise logging will occur at level INFO.')
	parser.add_argument('--mode', metavar='Type_String', type=str, nargs="?", default='rf', help='Use "mlp" to train a multilayer perceptron binary classifier, "ert" to train an extremly randomized trees classifier, "sgd" to train a SGDClassifier')
	parser.add_argument('--op', metavar='Type_string', type=str, nargs='?', default='sv', help='Use "scv" to perform a single cross-validation using TimeSeriesSplit. Use "rcv" to perform randomized grid search with cross validation. Use "sv" to perform a single training pass then save. Use "lc" to plot a learning curve with cross validation. Use "val" to perform a single validation on a saved model, specify model file to use with --modelfile argument.')
	parser.add_argument('--datafile', metavar='Type_string', type=str, nargs='?', default='data/traintest.csv', help='Datafile to use. Defaults to "data/traintest.csv". Use train and test file for st, sv, lc operations. Use validation file for val operation.')
	parser.add_argument('--modelfile', metavar='Type_string', type=str, nargs='?', default='', help='Model file to use. Required for val op.')
	parser.add_argument('--restrict_data', action='store_true', help='Use this argument to restrict the number of data lines used (for testing).')
	parser.add_argument('--rcviter', metavar='Type_string', type=str, nargs='?', default='10', help='Number of iterations to perform when using randomized grid search. Defaults to 10.')

	args = parser.parse_args()
	verbose = args.verbose
	mode = args.mode
	op = args.op
	restrict_data = args.restrict_data
	datafile = args.datafile
	model_file = args.modelfile
	rcv_iter = int(args.rcviter)

	# check arguments
	if mode not in ['ert', 'rf', 'mlp', 'sgd']:
		logging.critical('Mode {} not implemented. Quitting...'.format(mode))
		quit()
	if op not in ['sv', 'scv', 'rcv', 'lc', 'val']:
		logging.critical('Operation {} not implemented. Quitting...'.format(op))
		quit()
	try:
		if(not os.path.isfile(datafile)):
			logging.critical('Data file: {} not found. Quitting...'.format(datafile))
			quit()
	except TypeError:
		logging.critical('Invalid definition file given. Quitting...')
		quit()
	if op == 'val':
		try:
			if(not os.path.isfile(model_file)):
				logging.critical('Model file: {} not found. Quitting...'.format(model_file))
				quit()
		except TypeError:
			logging.critical('Invalid model file given. Quitting...')
			quit()

	save_timestamp = datetime.now().strftime('%Y%m%d-%H%M')

	# Logger
	print('Configuring logger...')
	if verbose == True :
		ll = logging.DEBUG
	else:
		ll = logging.INFO
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

	logging.info('Using mode: {} and performing operation: {}'.format(mode, op))
	
	# Get the data
	logging.debug('Obtaining data...')
	features, targets = data(datafile, restrict_data=restrict_data).get_data()
	logging.debug('Obtained {} samples for features and {} samples for targets'.format(len(features), len(targets)))

	# Execute
	o = operation_mode(mode)
	if op == 'sv':
		o.single_train_save(save_timestamp, features, targets)
	elif op == 'scv':
		o.single_cross_validation(save_timestamp, features, targets)
	elif op == 'lc':
		o.plot_learning_curve(save_timestamp, features, targets)
	elif op == 'val':
		o.valid(save_timestamp, model_file, features, targets)
	elif op == 'rcv':
		o.random_cross_validation(save_timestamp, features, targets, rcv_iter)
	quit()
