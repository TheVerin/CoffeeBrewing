import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import warnings
warnings.filterwarnings('ignore')

#Second grid search
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators': np.arange(10, 200, 20),
              'max_depth':[3,4,5,6],
              'learning_rate':[0.05, 0.1, 0.15, 0.5]}
BestParams2 = GridSearchCV(XGB, param_grid = parameters, scoring = 'accuracy', cv = 10)
BestParams2.fit(X_train[values], Y_train)
bests = BestParams2.best_params_
print('Best score:',BestParams2.best_score_)
print('Best parameters:', BestParams2.best_params_)

XGB = XGBClassifier(**BestParams2.best_params_)
XGBcv = cross_val_score(XGB, X_train[values], Y_train, cv = 10, scoring = 'accuracy' )
print('Accuracy:', XGBcv.mean())
print('Stability:', XGBcv.std()*100/XGBcv.mean())

XGBFinal = XGB.fit(X_train[values], Y_train)
XGBPred = XGB.predict(X_test[values])

#Checking accuracy
from sklearn.metrics import accuracy_score
Acc = accuracy_score(Y_test, XGBPred)
print ('Final Accuracy:', Acc)