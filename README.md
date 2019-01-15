# Delai

## Use machine learning to predict if the medication turnaround time will be short (<n min) or long (>n min)

### Description

Delai is a machine learning classifier that predicts if the medication turnaround time will be < or > n minutes given a set of input features. The exact cutoff can be tuned by modifying parameters in the program file.

### Motivation

The intended use of this model is to augment the usefulness of medication tracking systems used in hospital pharmacies to add a prediction of the turnaround time in addition to the tracking of the drug.

## Files

### delai.py

TODO

### preprocess.py

This file needs to be locally developed to interface the output of pharmacy databases with the machine learning program. The preprocess_features function within that file should return two pandas dataframeThe following columns should be provided by this file:

Features:
- drug_id: unique drug identifier
- user: user id who performed the operation
- time_frax: the time at which the operation was performed, provided as a float such that for example: 4:15 PM would be 16.25
- operation_code: pharmacy operation code. Examples include: new order, dispensation on existing order
- internal_or_external: code discriminating orders destined for inpatients or outpatients
- workbench: workbench id discriminating the type of drug preparation required. Examples include: sterile compounding, hazardous drug, commercially available drug
- workload: the rolling count of operations performed in the last 45 minutes
- Other columns should be assigned to the pharmacy schedule. Each column of this type should correspond to a scheduled role. Values in these columns should be the identifier of each person scheduled at that role at the time the operation was performed.

Targets:
- delay: a measure of the time between tracking initiation and the drug leaving the pharmacy

An example file is provided in preprocess_example.py

### Prerequisites

Developed using Python 3.7

Requires:

- Numpy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Scikit-plot
- Tensorflow
- Keras
- Joblib

## Contributors

Maxime Thibault.

## License

To be determined.