import pandas as pd
import numpy as np
import xgboost as xgb
from estimators import Estimator, DataSet, DataBase
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


#####SVM Hyperparameter space#####
kernel = ['rbf','linear','poly','sigmoid','precomputed'}
C = [0.5,1.0, 1.5, 2.0, 2.5, 3.0]
degree = np.arange(2,10)
shrinking = [True, False]
tol = [1e-5,1e-4,1e-3,1e-2,1e-1]
class_weight = [None, 'balanced']
max_iter = np.arange(0,1e3+1,1e2)

#####Logistic Regression Hyperparameter Space######


def genParams(baseList, blender, P, N):
	numBaseModels = len(baseList)
	mu = np.random.dirichlet(P)
	numModelsList = N * mu
	for i in np.arange(len(numModelsList)):
		if baseList[i] == 'SVM':
			for j in np.arange(numModelsList[i]):
				#randomly sample each hyperparameter - C, kernel type, degree, gamma, coef0, shrinking, tol, cache_size, class_weight, max_iter, decision_function_shape, random_state
				#probability parameter should always be true
				
		elif baseList[i] == 'LR':
			for j in np.arange(numModelsList[i]):
				#randomly sample each hyperparameter - 
		elif baseList[i] == 'RF':
			for j in np.arange(numModelsList[i]):
				parameters = {}
		else:
	return numModelsList,parameterListBaseModels, parameterListBlenderModel 

def Blend():
	

	
