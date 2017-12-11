import pandas as pd
import numpy as np
import xgboost as xgb
from estimators import Estimator, DataSet, DataBase
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def genParams(baseList, blender, P, N):
	numBaseModels = len(baseList)
	mu = np.random.dirichlet(P)
	numModelsList = N * mu
	#hyperparameters for the model
	#hyperparameters for belending model
	return numModelsList

def Blend():
	


