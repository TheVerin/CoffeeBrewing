import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import warnings
warnings.filterwarnings('ignore')

#First grid search
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators': np.arange(10, 200, 20),
              'max_depth':[3,4,5,6],
              'learning_rate':[0.05, 0.1, 0.15, 0.5]}
BestParams1 = GridSearchCV(XGB, param_grid = parameters, scoring = 'accuracy', cv = 10)
BestParams1.fit(X_train, Y_train)
bests = BestParams1.best_params_
print('Best score:',BestParams1.best_score_)
print('Best parameters:', BestParams1.best_params_)

XGB = XGBClassifier(**BestParams1.best_params_)