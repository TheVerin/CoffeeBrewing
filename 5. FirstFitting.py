import warnings
warnings.filterwarnings('ignore')


#Splitting for train and test split
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.15, random_state = 0)


#Fitting classifiers
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

XGB = XGBClassifier()
RanFor = RandomForestClassifier()

from sklearn.cross_validation import cross_val_score
XGBcv = cross_val_score(XGB, X_train, Y_train, cv = 10, scoring = 'accuracy' )
print('Accuracy:', XGBcv.mean())
print('Stability:', XGBcv.std()*100/XGBcv.mean())

RanForcv = cross_val_score(RanFor, X_train, Y_train, cv =10, scoring = 'accuracy')
print('Accuracy:',RanForcv.mean())
print('Stability',RanForcv.std()*100/RanForcv.mean())