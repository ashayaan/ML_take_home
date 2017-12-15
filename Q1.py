import random
import pandas as pd
import numpy as np
import xgboost as xgb
import sys

#from estimators import Estimator, DataSet, DataBase

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from sklearn import datasets
from itertools import chain

####This algorithm inspired from Feature Weighted Linear Stacking uses three base algorithms:
#   * Support Vector Machines
#   * Logistic Regression
#   * Random Forests
####The blending algorithm can be a random forest or a boosted tree

#####SVM Hyperparameter Space#####
kernel = ['rbf','linear','poly','sigmoid']
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
max_iter_lr = np.arange(1,1e3+1,1e2)
fit_intercept = [True, False]
intercept_scaling = [1,2,3,4,5]

#####Random Forest Classifier Hyperparameter Space#####
n_estimators = np.arange(10,51,10)
criterion = ['gini','entropy']
max_features = [None,'auto', 'sqrt', 'log2']
max_depth = [None,1,2,3,4,5]
min_samples_split = np.arange(2,11,2)
min_samples_leaf = np.arange(1,11,2)
max_leaf_nodes = [None,10,20,30,40,50,100]
min_impurity_decrease = np.arange(0.0,1.0,0.1)
bootstrap = [True,False]
oob_score = [True,False]

#Utility function to choose random hyperparameters in the hyperparameter space for the base learning algorithms
#Returns model object with the chosen parameters
def params(algorithm):
	if (algorithm == 'SVM'):
		#randomly sample each hyperparameter - C, kernel type, degree, gamma, coef0, shrinking, tol, cache_size, class_weight, max_iter
		#probability parameter should always be true
		k = str(np.random.choice(kernel))
		c = np.random.choice(C)
		d = np.random.choice(degree)
		s = np.random.choice(shrinking)
		t = np.random.choice(tol)
		cw = np.random.choice(class_weight)
		m = np.random.choice(max_iter)
		return svm.SVC(C=c,kernel=k,degree=d,shrinking=s,probability=True,tol=t,class_weight=cw,max_iter=m)
	elif (algorithm == 'LogisticRegression'):
		#randomly sample each hyperparamteter for the logistic regression classifier: penalty, dual,solver,fit_itercept,intercept_scaling, class_weight,C
		#handles dependencies between parameters. For example, for solvers, newron-cg, lbfgs, sag, the penalty is always l2, whereas for liblinear solver l1 or l2 penalty
		#chosen randomly
		s = np.random.choice(solver)
		if (s == 'newton-cg' or s == 'lbfgs' or s == 'sag'):
			p = 'l2'
			mc = np.random.choice(multi_class)
		elif (s == 'sag'):
			p = np.random.choice(penalty)
			mc = np.random.choice(multi_class)
		else:
			p = np.random.choice(penalty)
			mc = 'ovr'
		if (s == 'liblinear' and p == 'l2'):
			d = np.random.choice(dual)
		else:
			d = False
		f = np.random.choice(fit_intercept)
		i_s = np.random.choice(intercept_scaling)
		cw = np.random.choice(class_weight)
		t = np.random.choice(tol)
		mi = np.random.choice(max_iter_lr)
		c = np.random.choice(C)
		return LogisticRegression(penalty=p,dual=d,C=c,tol=t,fit_intercept=f,intercept_scaling=i_s,class_weight=cw,solver=s,max_iter=mi,multi_class=mc)
	elif (algorithm == 'RandomForest'):
		#randomly sample each hyperparameter for the random forest classifier from the hyperparameter space: n_estimators, criterion, max-features, max_depth, min_samples_split, 
		#min_samples_leaf, max_leaf_nodes, max_impurity_decrease, bootstrap and oob_score
		n_est = np.random.choice(n_estimators)
		cr = np.random.choice(criterion)
		max_f = np.random.choice(max_features)
		max_d = np.random.choice(max_depth)
		min_s_s = np.random.choice(min_samples_split)
		min_s_l = np.random.choice(min_samples_leaf)
		max_ln = np.random.choice(max_leaf_nodes)
		min_i_dec = np.random.choice(min_impurity_decrease)
		bs = np.random.choice(bootstrap)
		if bs == True:
			os = np.random.choice(oob_score)
		else:
			os = False
		cw = np.random.choice(class_weight)
		return RandomForestClassifier(n_estimators=n_est,criterion=cr,max_features=max_f,max_depth=max_d,min_samples_split=min_s_s,min_samples_leaf=min_s_l,max_leaf_nodes=max_ln,min_impurity_decrease= min_i_dec,bootstrap=bs,oob_score=os,class_weight=cw)

def genParams(baseList, blender, P, N):
	numBaseModels = len(baseList)
	mu = np.random.dirichlet(P)
	numModelsList = np.round((N * mu)).astype(int)
	baseListN =map(lambda i : [baseList[i]]*numModelsList[i], np.arange(numBaseModels))
	baseListSeries = pd.Series(list(chain.from_iterable(baseListN)))
	parametersBaseModels = baseListSeries.apply(params)
	#Choosing hyperparameters for the blender model
	if (blender == 'RandomForest'):
		### Hyperparameter sampling for random forest classifier ###
		n_est = np.random.choice(n_estimators)
		cr = np.random.choice(criterion)
		max_f = np.random.choice(max_features)
		max_d = np.random.choice(max_depth)
		cw = np.random.choice(class_weight)
		blenderModel = RandomForestClassifier(n_estimators=n_est,criterion=cr,max_features=max_f,max_depth=max_d,class_weight=cw)
	elif (blender == 'BoostedTree'):
		### Hyperparameter sampling for boosted trees using xgboost ###
		bs = np.random.rand(1)[0]
		md = random.randint(4,10)
		lr =  np.random.rand(1)[0]
		n_est = np.random.choice(np.arange(100,151,10))
		maxdelstep = np.random.choice(np.arange(1,11,1))
	 	blenderModel =  xgb.XGBClassifier(base_score=bs, learning_rate=lr, max_depth=md, n_estimators =n_est, max_delta_step=maxdelstep)
	return numModelsList,parametersBaseModels,blenderModel 

### Expanding dataset with results of learnt models ###
def expandFeatures(X, phi):
	Dfw = X.copy()
	X_u = X.loc[:,X.columns != 'label']
	M = phi.apply(lambda x: x.predict_proba(X_u))
	G = phi.apply(lambda x: x.predict(X_u))
	for i in np.arange(len(M)):
		for j in X_u.columns:
			for k in np.arange(M[i].shape[1]):
				### Compute feature weighted probabilities ###
				feature_weighted_prob_vector = X_u[j] * M[i][:,k]
				### Concatenate feature weighted probabilities to original features ###
				Dfw = pd.concat([Dfw,pd.DataFrame(feature_weighted_prob_vector)],axis=1)
		### Concatenate class prediction probabilities to original features ###
		Dfw = pd.concat([Dfw,pd.DataFrame(M[i])],axis=1)
		### Concatenate class predictions to original features ###
		Dfw = pd.concat([Dfw,pd.DataFrame(G[i])],axis=1)
	Dfw_new = Dfw.loc[:,Dfw.columns != 'label']
	Dfw_new.columns = np.arange(Dfw_new.shape[1])
	Dfw = pd.concat([Dfw_new,Dfw['label']],axis=1)
	### return new dataset after adding all the new features ###	
	return Dfw


def Blend(baseList, blender, dataset, testset, L, phi, N, psi):
	rho = np.random.uniform()	
	while (rho <= 0.1):
		rho = np.random.uniform()	
	dataset = pd.DataFrame(dataset)
	for l in np.arange(L):
		data_subset = dataset.sample(frac=rho,replace=True)
		y = data_subset['label']
		X = data_subset.drop(['label'],axis=1)
		### Train base models with data sample
		phi.apply(lambda x : x.fit(X,y))
		data_subset_complement = dataset.drop(data_subset.index,axis=0).reset_index(drop=True)
		### Adding features to the original dataset
		Dfw = expandFeatures(data_subset_complement,phi)
		labels = Dfw['label']
		data = Dfw.loc[:,Dfw.columns != 'label']
	#Fitting the data with the new features to the ensemble model
	if (blender == 'RandomForest'):
		psi.fit(data, labels)
	elif (blender == 'BoostedTree'):
		testset = expandFeatures(testset,phi)
		test_data = testset.loc[:,testset.columns!='label']
		test_label = testset['label']
		psi.fit(data, labels,eval_set=[(test_data,test_label)],eval_metric="merror",verbose=False)
	return psi


def blendingEnsemble(baseList, blender, dataset, P, N, k):
	#iris = datasets.load_iris()
	R = 10
	L = 2
	model_list = []
	for i in np.arange(R):
		print ("---------------------Iteration Number: "+str(i+1)+"-----------------------------")
		a, b, c =  genParams(baseList,blender,P,N)
		### K-Fold Cross Validation with k splits ###
		kf = StratifiedKFold(n_splits=k)
		error_list = []
		for train,test in kf.split(dataset.loc[:,irisdf.columns != 'label'],dataset['label']):
			trainset = dataset.iloc[train].reset_index(drop=True)
			testset = dataset.iloc[test].reset_index(drop=True)
			m  =  Blend(baseList,blender,trainset,testset,L,b,a,c)
			test = expandFeatures(testset,b)
			### Compute error on the test set ###
			if (blender == 'RandomForest'):
				test_data = test.loc[:,test.columns != 'label']
				test_labels = test['label']
				error = 1 - m.score(test_data,test_labels)
				error_list = np.append(error_list,error)
			elif (blender == 'BoostedTree'):
				error_list.append(np.mean(m.evals_result()['validation_0']['merror']))
		### Mean error on all cross-validation sets ###	
		avg_er = np.mean(error_list)
		model_list.append((avg_er,a,b,c))
	### Get model with minimum error ###	
	model_final =  min(model_list,key=lambda t: t[0])
	model_blended_final = Blend(baseList,blender,dataset,dataset,L,model_final[2],model_final[1],model_final[3])
	test = expandFeatures(dataset,model_final[2])
	if (blender == 'RandomForest'):
		print ("Final error using random forest blend model: "+str(1.00 - model_blended_final.score(test.loc[:,test.columns !='label'],test['label'])))
	elif (blender == 'BoostedTree'):
		err = model_blended_final.evals_result()['validation_0']['merror']
		error = np.mean(err)
		print ("Final error with xgboost blend model: " + str(error))
	return model_blended_final

if __name__ == "__main__":
	iris = datasets.load_iris()    ### Sample dataset used: iris dataset ###
	irisdf = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['label'])
	baseList = ['SVM','LogisticRegression','RandomForest']  ### Three base algorithms implemented: SVM, Logistic Regression, Random Forest ###
	blender = sys.argv[1]   ### Set blender = 'BoostedTree' to use xgboost as blending algorithm ###
	P = [0.1,0.5,0.4]          ### Preference array ###
	N = 7			   ### Total No of models N ###
	k = 5                      ### Number of cross validation sets ###
	model = blendingEnsemble(baseList, blender, irisdf, P, N, k) ### final model using blender trained on complete dataset ###	

