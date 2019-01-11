import logging

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


class data:

	def __init__(self, features_file='data/features.pkl', targets_file='data/targets.pkl'):
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


def make_data_graphs(features, targets):
	all = pd.concat([features, targets], axis=1)

	fig = sns.catplot(x='fait_a_la_fab', y='delai', kind='box', data=all, showfliers=False)
	plt.show()

	quit()


def make_training_graph():
	training_datafile = 'model/20190104-1612history.csv'
	data = pd.read_csv(training_datafile, index_col=0, header=0).stack().reset_index()
	data.rename(inplace=True, index=str, columns={'level_1':'metric', 0:'value'})
	fig = sns.relplot(kind='line', x='epoch', y='value', hue='metric', data=data)
	plt.show()

if __name__ == '__main__':

	# configure environment
	print('Configuring logger...')

	# Logger
	logging.basicConfig(
		level=logging.DEBUG,
		format="%(asctime)s [%(levelname)s]  %(message)s",
		handlers=[
			logging.StreamHandler()
		])
	logging.debug('Logger successfully configured.')

	features, targets = data().get_data()
	make_data_graphs(features, targets)
