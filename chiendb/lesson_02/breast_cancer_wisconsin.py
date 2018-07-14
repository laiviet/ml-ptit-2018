import os
import requests
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score


def load_data():
	save_file_name = 'breast_cancer_wisconsin.csv'
	if os.path.isfile(save_file_name):
		data = np.genfromtxt(save_file_name, delimiter=',')
	else:
		r = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data')
		data = r.text.split('\n')
		data = [x for x in data if x != '' and '?' not in x]
		data = np.array([list(map(int, x.split(','))) for x in data])
		np.savetxt(save_file_name, data, delimiter=',')

	return data[:, 1: -1], data[:, -1]


def sigmoid(x):
	return 1/(1+np.exp(-x))


def sigmoid_derivative(x):
	return x*(1-x)


if __name__ == '__main__':
	X, y = load_data()
	y[y==2], y[y==4] = 0, 1
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	num_steps = 30000
	num_samples = np.shape(X_train)[0]
	num_features = np.shape(X_train)[1]
	num_classes = 2
	threshold = 0.5
	learning_rate = 0.001
	y_train = y_train.reshape((num_samples, 1))
	W1 = np.random.rand(num_features, 20)
	W2 = np.random.rand(20, 1)
	for t in range(num_steps):
		Z1 = np.dot(X_train, W1)
		L1 = sigmoid(Z1)
		y_hat = sigmoid(np.dot(sigmoid(np.dot(X_train, W1)), W2))
		error = y_hat - y_train
		error_Z2 = error*sigmoid_derivative(y_hat)
		error_W1 = np.dot(X_train.T, np.dot(error_Z2, W2.T)*sigmoid_derivative(L1))
		error_W2 = np.dot(L1.T, error_Z2)
		W1 -= learning_rate*error_W1
		W2 -= learning_rate*error_W2
		if (t+1) % 1000 == 0:
			y_pred = sigmoid(np.dot(sigmoid(np.dot(X_test, W1)), W2)).reshape((1, len(y_test)))[0]
			y_pred[y_pred > threshold] = 1
			y_pred[y_pred <= threshold] = 0
			acc = accuracy_score(y_test, y_pred)
			loss = np.sum((y_pred-y_test)*(y_pred-y_test))/(2*len(y_test))
			print('Step = {}, loss = {}, Acc = {}'.format(t+1, loss, acc))


		
		

