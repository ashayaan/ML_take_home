import pandas as pd
import numpy as np
import xgboost as xgb
from estimators import Estimator, DataSet, DataBase
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from itertools import chain

#####SVM Hyperparameter Space#####
kernel = ['rbf','linear','poly','sigmoid','precomputed']
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

#####Random Forest Classifier Hyperparameter Space#####
n_estimators = np.arange(10,51,10)
criterion = ['gini','entropy']
max_features = [None,'auto', 'sqrt', 'log2']
max_depth = [None,1,2,3,4,5]


def params(algorithm):
	if (algorithm == 'SVM'):
		#randomly sample each hyperparameter - C, kernel type, degree, gamma, coef0, shrinking, tol, cache_size, class_weight, max_iter
		#probability parameter should always be true
		k = np.random.choice(kernel) 
		c = np.random.choice(C)
		d = np.random.choice(degree)
		s = np.random.choice(shrinking)
		t = np.random.choice(tol)
		cw = np.random.choice(class_weight)
		m = np.random.choice(max_iter)
		return {'C':c, 'kernel':k,'degree':d,'shrinking':s,'probability':True,'tol':t,'class_weight':cw, 'max_iter':m}	
	elif (algorithm == 'LogisticRegression'):
		p = np.random.choice(penalty)
		d = np.random.choice(dual)
		s = np.random.choice(solver)
		mc = np.random.choice(multi_class)
		f = np.random.choice(fit_intercept)
		i_s = np.random.choice(intercept_scaling)
		cw = np.random.choice(class_weight)
		t = np.random.choice(tol)
		mi = np.random.choice(max_iter)
		c = np.random.choice(C)
		return {'penalty':p,'dual':d,'tol':t,'C':c,'fit_itercept':f,'intercept_scaling':i_s, 'class_weight':cw,'solver':s,'max_iter':mi,'multi_class':mc}
	elif (algorithm == 'RandomForest'):
		n_est = np.random.choice(n_estimators)
		cr = np.random.choice(criterion)
		max_f = np.random.choice(max_features)
		max_d = np.random.choice(max_depth)
		cw = np.random.choice(class_weight)
		return {'n_estimators':n_est,'criterion':cr,'max_features':max_f,'max_depth':max_d,'class_weight':cw}

def genParams(baseList, blender, P, N):
	numBaseModels = len(baseList)
	mu = np.random.dirichlet(P)
	numModelsList = np.round((N * mu)).astype(int)
	baseListN =map(lambda i : [baseList[i]]*numModelsList[i], np.arange(numBaseModels))
	baseListSeries = pd.Series(list(chain.from_iterable(baseListN)))
	parametersBaseModels = baseListSeries.apply(params)
	#BlendingModels = {'RandomForest','BoostedTree'}
	#blender = np.random.choice(BlendingModels)
	if (blender == 'RandomForest'):
		#Hyperparameter sampling for random forest classifier
		n_est = np.random.choice(n_estimators)
		cr = np.random.choice(criterion)
		max_f = np.random.choice(max_features)
		max_d = np.random.choice(max_depth)
		cw = np.random.choice(class_weight)
		parameterListBlenderModel = {'n_estimators':n_est,'criterion':cr,'max_features':max_f,'max_depth':max_d,'class_weight':cw}
	elif (blender == 'BoostedTree'):
		parameterListBlenderModel = {'To be defined':None}
		#Hyperparameter sampling for boosted trees using xgboost
	#for i in np.arange(len(numModelsList)):
	#	if baseList[i] == 'SVM':
	#		for j in np.arange(numModelsList[i]):
	#			#randomly sample each hyperparameter - C, kernel type, degree, gamma, coef0, shrinking, tol, cache_size, class_weight, max_iter
	#			#probability parameter should always be true
	#			k = np.random.choice(kernel)
	#			c = np.random.choice(C)
	#			d = np.random.choice(degree)
	#			s = np.random.choice(shrinking)
	#			t = np.random.choice(tol)
	#			cw = np.random.choice(class_weight)
	#			m = np.random.choice(max_iter)
	#	elif baseList[i] == 'LR':
	#		for j in np.arange(numModelsList[i]):
	#			#randomly sample each hyperparameter - penalty, dual, solver, multi_class, fit_intercept, intercept_scaling, class_weight, max_iter, C, tol
	#			
	#	elif baseList[i] == 'RF':
	#		for j in np.arange(numModelsList[i]):
	#			parameters = {}
	#	else:
	return numModelsList,parametersBaseModels, parameterListBlenderModel 


def Blend(baseList, blender, dataset, L, phi, N, psi):
	rho = np.random.uniform()	
	for l in np.arange(L):
		dataset = 
	

#Test genParams
#a, b, c =  genParams(['SVM','LogisticRegression','RandomForest'],'RandomForest',[0.1,0.5,0.4],6)


	
