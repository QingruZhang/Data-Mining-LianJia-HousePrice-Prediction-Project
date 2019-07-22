import re
import sqlite3
import pickle
import unicodecsv as csv
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.naive_bayes import GaussianNB


def data_provider_with_processer(preprocessing="None", selected_y = 1):
	X_train, X_test, Y_train, Y_test = provider(is_regression=True)
	# choose selected_y = 1, to select unit price as y
	if selected_y:
		Y_train = Y_train[:,selected_y]
		Y_test = Y_test[:,selected_y]

	# Normalize the label 
	# salary_max, salary_min = np.max(Y_train), np.min(Y_train)
	# Y_train = (Y_train - salary_min) / float(salary_max - salary_min)
	# Y_test = (Y_test - salary_min) / float(salary_max - salary_min)
	if preprocessing == "normalize":
		normalizer = Normalizer()
		X_train = normalizer.fit_transform(X_train)
		X_test = normalizer.fit_transform(X_test)
	elif preprocessing == "minmax":
		minmaxscaler = MinMaxScaler()
		X_train = minmaxscaler.fit_transform(X_train)
		X_test = minmaxscaler.fit_transform(X_test)
	elif preprocessing == "standard":
		standardscale = StandardScaler()
		X_train = standardscale.fit_transform(X_train)
		X_test = standardscale.fit_transform(X_test)
	else:
		pass
	return X_train, X_test, Y_train, Y_test


def provider(filepath="data/csv/final_training_data.csv", 
			 is_regression=True,):
	# read data from the csv file 
	df_all = pd.read_csv(filepath, encoding = 'gbk')
	data = df_all.iloc[:, 2:]
	df_y_valuse = df_all.iloc[:, :2]

	# split data
	X_train, X_test, Y_train, Y_test = train_test_split(data.values, df_y_valuse.values, test_size=0.2, random_state=42)

	return X_train, X_test, Y_train, Y_test


if __name__ == '__main__':
	filepath="data/final_training_data.csv"
	df_all = pd.read_csv(filepath, encoding = 'gbk')
	data = df_all.iloc[:,2:]
	df_y_valuse = df_all.iloc[:,0:2]
	X_train, X_test, Y_train, Y_test = train_test_split(data.values, df_y_valuse.values, test_size=0.2, random_state=42)