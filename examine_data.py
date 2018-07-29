import math

import pandas as pd
from matplotlib import pyplot as plt

def linear_scale(series):
	min_val = series.min()
	max_val = series.max()
	scale = (max_val - min_val) / 2.0
	return series.apply(lambda x:((x - min_val) / scale) - 1.0)

def log_normalize(series):
	return series.apply(lambda x:math.log(x+1.0))

features_file = 'features.pkl'
all_features = pd.read_pickle(features_file)
targets_file = 'targets.pkl'
all_targets = pd.read_pickle(targets_file)

#make sure data loaded properly
print('Processed features summary:')
print(all_features.describe())
print(all_features.head())

print('Processed targets summary:')
print(all_targets.describe())
print(all_targets.head())

all_features['charge_de_travail_endroit']
all_features['charge_de_travail_fabounon']
all_features['charge_de_travail_saisie']
all_features['normalized_charge_de_travail_endroit'] = linear_scale(log_normalize(all_features['charge_de_travail_endroit']))
all_features['normalized_charge_de_travail_fabounon'] = linear_scale(log_normalize(all_features['charge_de_travail_fabounon']))
all_features['normalized_charge_de_travail_saisie'] = linear_scale(log_normalize(all_features['charge_de_travail_saisie']))
all_features['normalized_squared_charge_de_travail_endroit'] = linear_scale(log_normalize(all_features['charge_de_travail_endroit'] **0.5))
all_features['normalized_squared_charge_de_travail_fabounon'] = linear_scale(log_normalize(all_features['charge_de_travail_fabounon'] **0.5))
all_features['normalized_squared_charge_de_travail_saisie'] = linear_scale(log_normalize(all_features['charge_de_travail_saisie'] **0.5))

statut = all_features.ORDO_STATUT.unique()
endroit = all_features.endroit.unique()
oper = all_features.SORE_CODE_OPER.value_counts()

usagers = all_features.SORE_USAGER.value_counts()
seuil = 10
usagers_filtre = usagers[usagers >= seuil].index.tolist()

medicaments = all_features['MEDI_NOM'].value_counts()
seuil = 100
medicaments_filtre = medicaments[medicaments >= seuil].index.tolist()

fx_list = ['at_saisie1', 'at_saisie2', 'at_fab1', 'at_fab2', 'pharm1', 'pharm2', 'pharmf']
stacked = all_features[fx_list].stack(dropna=True)
counts = pd.value_counts(stacked)
counts = counts.drop('')
seuil = 200
personnes_filtre = counts[counts >= seuil]
personnes_list = counts[counts >= seuil].index.tolist()

all_targets['delai_cat'] = all_targets['delai'].apply(lambda x: 1 if x < 30 else (2 if x < 60 else (3 if x < 90 else (4))))

at_saisie1 = all_features['at_saisie1'].value_counts()
at_saisie1 = at_saisie1.filter(items=personnes_list)
at_saisie2 = all_features['at_saisie2'].value_counts()
at_saisie2 = at_saisie2.filter(items=personnes_list)
at_fab1 = all_features['at_fab1'].value_counts()
at_fab1 = at_fab1.filter(items=personnes_list)
at_fab2 = all_features['at_fab2'].value_counts()
at_fab2 = at_fab2.filter(items=personnes_list)
pharm1 = all_features['pharm1'].value_counts()
pharm1 = pharm1.filter(items=personnes_list)
pharm2 = all_features['pharm2'].value_counts()
pharm2 = pharm2.filter(items=personnes_list)
pharmf = all_features['pharmf'].value_counts()
pharmf = pharmf.filter(items=personnes_list)

bob, ax = plt.subplots(3,5)
ax[0,0].hist(usagers_filtre)
ax[0,0].set_title('usagers')
ax[0,1].hist(medicaments_filtre)
ax[0,1].set_title('medics')
ax[0,2].hist(statut)
ax[0,2].set_title('statuts')
ax[0,3].hist(endroit)
ax[0,3].set_title('endroits')
ax[0,4].hist(at_saisie1)
ax[0,4].set_title('at_saisie1')
ax[1,0].hist(at_saisie1)
ax[1,0].set_title('at_saisie2')
ax[1,1].hist(at_saisie2)
ax[1,1].set_title('at_fab1')
ax[1,2].hist(at_fab1)
ax[1,2].set_title('at_fab2')
ax[1,3].hist(at_fab2)
ax[1,3].set_title('pharm1')
ax[1,4].hist(pharm1)
ax[1,4].set_title('pharm2')
ax[2,0].hist(pharm2)
ax[2,0].set_title('pharmf')
ax[2,1].hist(oper)
ax[2,1].set_title('codes oper')

plt.show()

f, hist = plt.subplots(3)
hist[0].hist(all_features['unixdateheure'])
hist[0].set_title('Date et heure unix')
hist[1].hist(linear_scale(all_features['unixdateheure']))
hist[1].set_title('Date et heure unix normalisée')
hist[2].hist(all_features['heure_frax'])
hist[2].set_title('Heure')
plt.show()

target_hist = plt.hist(all_targets['delai_cat'])
plt.show()

f, hist = plt.subplots(3,3)
hist[0,0].hist(all_features['charge_de_travail_endroit'])
hist[0,0].set_title('Charge de travail par endroit')
hist[0,1].hist(all_features['charge_de_travail_fabounon'])
hist[0,1].set_title('Charge de travail fab ou distrib')
hist[0,2].hist(all_features['charge_de_travail_saisie'])
hist[0,2].set_title('Charge de travail à la saisie')
hist[1,0].hist(all_features['normalized_charge_de_travail_endroit'])
hist[1,0].set_title('Charge de travail log-normalisée par endroit')
hist[1,1].hist(all_features['normalized_charge_de_travail_fabounon'])
hist[1,1].set_title('Charge de travail log-normalisée fab ou distrib')
hist[1,2].hist(all_features['normalized_charge_de_travail_saisie'])
hist[1,2].set_title('Charge de travail log-normalisée à la saisie')
hist[2,0].hist(all_features['normalized_squared_charge_de_travail_endroit'])
hist[2,0].set_title('Charge de travail au carré par endroit')
hist[2,1].hist(all_features['normalized_squared_charge_de_travail_fabounon'])
hist[2,1].set_title('Charge de travail au carré fab ou distrib')
hist[2,2].hist(all_features['normalized_squared_charge_de_travail_saisie'])
hist[2,2].set_title('Charge de travail au carré à la saisie')
plt.show()



