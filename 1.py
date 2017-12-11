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
max_iter = np.append(max_iter,-1)

#####Logistic Regression Hyperparameter Space######
penalty = ['l1','l2']
dual = [True, False]
solver = ['newton-cg','lbfgs','liblinear','sag','saga']
multi_class = ['ovr','multinomial']
fit_intercept = [True, False]
intercept_scaling = [1,2,3,4,5]

#####Random Forest Classifier hyperparameters#####
n_estimators = np.arange(10,51,10)
criterion = ['gini','entropy']
max_features = [None,'auto', 'sqrt', 'log2']
max_depth = [None,1,2,3,4,5]


def params(algorithm):
	if (algorithm == 'SVM'):
		#randomly sample each hyperparameter - C, kernel type, degree, gamma, coef0, shrinking, tol, cache_size, class_weight, max_iter
		#probability parameter should always be true
		k = np.rand.choice(kernel) 
		c = np.rand.choice(C)
		d = np.rand.choice(degree)
		s = np.rand.choice(shrinking)
		t = np.rand.choice(tol)
		cw = np.rand.choice(class_weight)
		m = np.rand.choice(max_iter)
		return {'C':c, 'kernel':k,'degree':d,'shrinking'=s,'probability':True,'tol'=t,'class_weight'=cw, 'max_iter'=m}	
	elif (algorithm == 'LogisticRegression'):
		p = np.rand.choice(penalty)
		d = np.rand.choice(dual)
		s = np.rand.choice(solver)
		mc = np.rand.choice(multi_class)
		f = np.rand.choice(fit_intercept)
		i_s = np.rand.choice(intercept_scaling)
		cw = np.rand.choice(class_weight)
		t = np.rand.choice(tol)
		mi = np.rand.choice(max_iter)
		c = np.rand.choice(C)
		return {}
	elif (algorithm == 'RandomForest'):


def genParams(baseList, blender, P, N):
	numBaseModels = len(baseList)
	mu = np.random.dirichlet(P)
	numModelsList = N * mu
	# baseListSeries = 
	parametersBaseModels = baseListSeries.apply(params)
	BlendingModels = {'RandomForest','BoostedTree'}
	blender = np.rand.choice(BlendingModels)
	if (blender == 'RandomForest'):
		#Hyperparameter sampling for random forest classifier
	elif (blender == 'BoostedTree'):
		#Hyperparameter sampling for boosted trees using xgboost
	#for i in np.arange(len(numModelsList)):
	#	if baseList[i] == 'SVM':
	#		for j in np.arange(numModelsList[i]):
	#			#randomly sample each hyperparameter - C, kernel type, degree, gamma, coef0, shrinking, tol, cache_size, class_weight, max_iter
	#			#probability parameter should always be true
	#			k = np.rand.choice(kernel)
	#			c = np.rand.choice(C)
	#			d = np.rand.choice(degree)
	#			s = np.rand.choice(shrinking)
	#			t = np.rand.choice(tol)
	#			cw = np.rand.choice(class_weight)
	#			m = np.rand.choice(max_iter)
	#	elif baseList[i] == 'LR':
	#		for j in np.arange(numModelsList[i]):
	#			#randomly sample each hyperparameter - penalty, dual, solver, multi_class, fit_intercept, intercept_scaling, class_weight, max_iter, C, tol
	#			
	#	elif baseList[i] == 'RF':
	#		for j in np.arange(numModelsList[i]):
	#			parameters = {}
	#	else:
	return numModelsList,parameterListBaseModels, parameterListBlenderModel 

def Blend():
	

	
