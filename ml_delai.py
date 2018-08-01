import logging
import math
import os

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import cm, gridspec
from matplotlib import pyplot as plt
from sklearn import metrics

####################
#### TWEAKABLES ####
####################

features_file = 'features.pkl'
targets_file = 'targets.pkl'

chunk_definition_dict = {'training_examples':0.79, 'validation_examples':0.01, 'test_examples':0.20}

learning_rate = 0.00003
batch_size = 3000
epochs = 300
hu = [20,10]

save_dir = '20180731_classifier'

####################
#### FUNCTIONS #####
####################

def chunkify_dataframe(df, chunk_definition_dict):
	nb_to_skip = 0
	nb_of_rows = df.shape[0]
	chunk_dict = {}
	df_dict = {}
	for chunk_name, chunk_proportion in chunk_definition_dict.items():
		chunk_size = math.floor(nb_of_rows * chunk_proportion)
		chunk_dict[chunk_name] = chunk_size
	for chunk, size in chunk_dict.items():
		df_dict[chunk] = df.iloc[nb_to_skip:nb_to_skip + size].copy()
		nb_to_skip += size
	return df_dict

def get_quantile_based_boundaries(feature_values, num_buckets):
	boundaries = np.arange(1.0, num_buckets) / num_buckets
	quantiles = feature_values.quantile(boundaries)
	return [quantiles[q] for q in quantiles.keys()]

def construct_feature_columns():
	"""Construct the TensorFlow Feature Columns.

	Returns:
		A set of feature columns
	""" 
	# layer 1 input columns
	heure_frax = tf.feature_column.numeric_column("heure_frax")
	charge_de_travail_endroit = tf.feature_column.numeric_column("charge_de_travail_endroit")
	charge_de_travail_fabounon = tf.feature_column.numeric_column("charge_de_travail_fabounon")
	charge_de_travail_saisie = tf.feature_column.numeric_column("charge_de_travail_saisie")
	fin_de_semaine = tf.feature_column.categorical_column_with_identity(key="fin_de_semaine", num_buckets=2)
	soir = tf.feature_column.categorical_column_with_identity(key="soir", num_buckets=2)
	endroit = tf.feature_column.categorical_column_with_identity(key="endroit", num_buckets=7)
	fait_a_la_fab = tf.feature_column.categorical_column_with_identity(key="fait_a_la_fab", num_buckets=2)
	oper = tf.feature_column.categorical_column_with_vocabulary_list(key="SORE_CODE_OPER", vocabulary_list=['NO', 'RP'])
	ORDO_STATUT = tf.feature_column.categorical_column_with_vocabulary_list(key="ORDO_STATUT", vocabulary_list=['H', 'E'])
	usager = tf.feature_column.categorical_column_with_hash_bucket(key="SORE_USAGER", hash_bucket_size=10)
	MEDI_NOM = tf.feature_column.categorical_column_with_hash_bucket(key="MEDI_NOM", hash_bucket_size=100)
	at_saisie1 = tf.feature_column.categorical_column_with_hash_bucket('at_saisie1', hash_bucket_size=10)
	at_saisie2 = tf.feature_column.categorical_column_with_hash_bucket('at_saisie2', hash_bucket_size=10)
	at_fab1 = tf.feature_column.categorical_column_with_hash_bucket('at_fab1', hash_bucket_size=10)
	at_fab2 = tf.feature_column.categorical_column_with_hash_bucket('at_fab2', hash_bucket_size=10)
	pharm1 = tf.feature_column.categorical_column_with_hash_bucket('pharm1', hash_bucket_size=10)
	pharm2 = tf.feature_column.categorical_column_with_hash_bucket('pharm2', hash_bucket_size=10)
	pharmf = tf.feature_column.categorical_column_with_hash_bucket('pharmf', hash_bucket_size=10)

	# layer 2 processed columns

	indicator_oper = tf.feature_column.indicator_column(categorical_column=oper)
	indicator_findesemaine = tf.feature_column.indicator_column(categorical_column=fin_de_semaine)
	indicator_soir = tf.feature_column.indicator_column(categorical_column=soir)
	indicator_ordostatut = tf.feature_column.indicator_column(categorical_column=ORDO_STATUT)
	indicator_faitalafab = tf.feature_column.indicator_column(categorical_column=fait_a_la_fab)
	indicator_endroit = tf.feature_column.indicator_column(categorical_column=endroit)

	indicator_usager = tf.feature_column.indicator_column(categorical_column=usager)
	indicator_MEDI_NOM = tf.feature_column.indicator_column(categorical_column=MEDI_NOM)
	indicator_at_saisie1 =tf.feature_column.indicator_column(categorical_column=at_saisie1)
	indicator_at_saisie2 =tf.feature_column.indicator_column(categorical_column=at_saisie2)
	indicator_at_fab1 = tf.feature_column.indicator_column(categorical_column=at_fab1)
	indicator_at_fab2 = tf.feature_column.indicator_column(categorical_column=at_fab2)
	indicator_pharm1 = tf.feature_column.indicator_column(categorical_column=pharm1)
	indicator_pharm2 = tf.feature_column.indicator_column(categorical_column=pharm2)
	indicator_pharmf = tf.feature_column.indicator_column(categorical_column=pharmf)

	bucketized_heure = tf.feature_column.bucketized_column(heure_frax, boundaries=get_quantile_based_boundaries(training_examples["heure_frax"], 24))
	bucketized_chargedetravailsaisie = tf.feature_column.bucketized_column(charge_de_travail_saisie, boundaries=get_quantile_based_boundaries(training_examples['charge_de_travail_saisie'], 10))
	bucketized_chargedetravailfabounon = tf.feature_column.bucketized_column(charge_de_travail_fabounon, boundaries=get_quantile_based_boundaries(training_examples['charge_de_travail_fabounon'], 10))
	bucketized_chargedetravailendroit = tf.feature_column.bucketized_column(charge_de_travail_endroit, boundaries=get_quantile_based_boundaries(training_examples['charge_de_travail_endroit'], 10))
	
	feature_columns = set([bucketized_heure, indicator_soir, indicator_findesemaine, bucketized_chargedetravailendroit, bucketized_chargedetravailfabounon, bucketized_chargedetravailsaisie, indicator_usager, indicator_oper, indicator_MEDI_NOM, indicator_ordostatut, indicator_faitalafab, indicator_endroit, indicator_at_saisie1, indicator_at_saisie2, indicator_at_fab1, indicator_at_fab2, indicator_pharm1, indicator_pharm2, indicator_pharmf])
	
	return feature_columns
	
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
	"""Trains a linear regression model.
	
	Args:
		features: pandas DataFrame of features
		targets: pandas DataFrame of targets
		batch_size: Size of batches to be passed to the model
		shuffle: True or False. Whether to shuffle the data.
		num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
	Returns:
		Tuple of (features, labels) for next data batch
	"""
	
	# Convert pandas data into a dict of np arrays.
	features = {key:np.array(value) for key,value in dict(features).items()}                                           
 
	# Construct a dataset, and configure batching/repeating.
	ds = tf.data.Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
	ds = ds.batch(batch_size).repeat(num_epochs)
	
	# Shuffle the data, if specified.
	if shuffle:
		ds = ds.shuffle(10000)
	
	# Return the next batch of data.
	features, labels = ds.make_one_shot_iterator().get_next()
	return features, labels

def train_model(
	learning_rate,
	steps,
	batch_size,
	training_examples,
	training_targets,
	validation_examples,
	validation_targets):
	"""Trains a linear regression model.
	
	In addition to training, this function also prints training progress information,
	as well as a plot of the training and validation loss over time.
	
	Args:
	learning_rate: A `float`, the learning rate.
	steps: A non-zero `int`, the total number of training steps. A training step
		consists of a forward and backward pass using a single batch.
	feature_columns: A `set` specifying the input feature columns to use.
	training_examples: A `DataFrame` containing one or more columns from
		`california_housing_dataframe` to use as input features for training.
	training_targets: A `DataFrame` containing exactly one column from
		`california_housing_dataframe` to use as target for training.
	validation_examples: A `DataFrame` containing one or more columns from
		`california_housing_dataframe` to use as input features for validation.
	validation_targets: A `DataFrame` containing exactly one column from
		`california_housing_dataframe` to use as target for validation.
		
	Returns:
	A `LinearRegressor` object trained on the training data.
	"""
	print('Steps to run: {}'.format(steps))

	# Create a linear regressor object.
	DNN_classifier = tf.estimator.DNNClassifier(hidden_units=hu, n_classes=2, feature_columns=feature_columns, optimizer=my_optimizer, model_dir=save_dir, config=my_checkpointing_config)
	
	training_input_fn = lambda: my_input_fn(training_examples, 
											training_targets["delai_cat"],
											num_epochs=epochs, 
											batch_size=batch_size)
	predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
													validation_targets["delai_cat"], 
													num_epochs=1, 
													shuffle=False)

	# Train the model, but do so inside a loop so that we can periodically assess
	# loss metrics.
	print("Training model...")
	# Train the model, starting from the prior state.
	train_spec = tf.estimator.TrainSpec(input_fn=training_input_fn, max_steps=steps)
	eval_spec = tf.estimator.EvalSpec(input_fn=predict_validation_input_fn, start_delay_secs=1800, throttle_secs=1200)
	tf.estimator.train_and_evaluate(DNN_classifier, train_spec, eval_spec)

	return DNN_classifier

def make_predictions(test_features, test_targets):
	DNN_classifier = tf.estimator.DNNClassifier(hidden_units=hu, n_classes=2, feature_columns=feature_columns, optimizer=my_optimizer, model_dir=save_dir, config=my_checkpointing_config)
	predict_test_input_fn = lambda: my_input_fn(test_features, 
													test_targets['delai_cat'], 
													num_epochs=1, 
													shuffle=False)
	predictions = DNN_classifier.predict(input_fn=predict_test_input_fn)
	output_string=''
	final_predictions = np.array([item['class_ids'][0] for item in predictions])
	for prediction, features, expected in zip(final_predictions, test_features.itertuples(), test_targets.itertuples()):
		if features.soir == 0:
			moment_string='jour'
		else:
			moment_string='soir'
		if features.fin_de_semaine == 0:
			fds_string='semaine'
		else:
			fds_string='fin de semaine'
		to_append='Médicament: {}   Moment: {}   Endroit: {}   Opération {}   AT saisie 1: {}   AT saisie 2: {}   AT fab 1: {}   AT fab 2: {}   Pharm 1: {}   Pharm 2: {}   Pharm f: {}, Prédit: {}   Réel {} \n'.format(features.MEDI_NOM, moment_string+ ' ' +fds_string, features.endroit, features.SORE_CODE_OPER, features.at_saisie1, features.at_saisie2, features.at_fab1, features.at_fab2, features.pharm1, features.pharm2, features.pharmf, prediction, expected.delai_cat)
		output_string += to_append
		with open('predictions.txt', mode='w', encoding='utf-8', errors='strict') as output_file:
			output_file.write(output_string)
		
	return final_predictions

####################
##### EXECUTE ######
####################

if __name__ == '__main__':

	# configure environment
	print('Configuring logger...')

	# TF logger
	log = logging.getLogger('tensorflow')
	log.setLevel(logging.INFO)

	# load data
	print('Loading data...')
	all_features = pd.read_pickle(features_file)
	all_features = all_features.drop(['unixdateheure'
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
	,'f17'], axis = 1)

	all_targets = pd.read_pickle(targets_file)

	all_targets['delai_cat'] = all_targets['delai'].apply(lambda x: 0 if x < 60 else 1)
	all_targets = all_targets.drop('delai', axis=1)

	# Chunkify data
	print('Chunkifying dataframe...')
	features_df = chunkify_dataframe(all_features, chunk_definition_dict)
	training_examples = features_df['training_examples']
	validation_examples = features_df['validation_examples']
	test_examples = features_df['test_examples']
	training_targets = all_targets.loc[training_examples.index]
	validation_targets = all_targets.loc[validation_examples.index]
	test_targets = all_targets.loc[test_examples.index]
	del all_features
	del all_targets

	steps = math.floor(training_examples.shape[0]/batch_size) * epochs

	feature_columns = construct_feature_columns()
	my_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
	my_checkpointing_config = tf.estimator.RunConfig(save_checkpoints_secs=3600, keep_checkpoint_max = 3)
	
	trained_model = train_model(
		learning_rate=learning_rate,
		steps=steps,
		batch_size=batch_size,
		training_examples=training_examples,
		training_targets=training_targets,
		validation_examples=validation_examples,
		validation_targets=validation_targets)
	
	final_predictions = make_predictions(validation_examples, validation_targets)

	  # Output a plot of the confusion matrix.
	cm = metrics.confusion_matrix(validation_targets, final_predictions)
	# Normalize the confusion matrix by row (i.e by the number of samples
	# in each class).
	cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
	ax = sns.heatmap(cm_normalized, cmap="bone_r")
	ax.set_aspect(1)
	plt.title("Confusion matrix")
	plt.ylabel("True label")
	plt.xlabel("Predicted label")
	plt.show()

	os.system('tensorboard --logdir=' + save_dir)
