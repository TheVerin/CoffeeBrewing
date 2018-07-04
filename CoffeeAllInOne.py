#How Tasty is Your Coffee?
#Business aim of this project is to get the most important features in caffe preparing process.
#It would be a multilabel classification problem (levels 0-5) so I will try with  non-linear models.
#So, Let;s go!

#IMporting libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import warnings
warnings.filterwarnings('ignore')

#Importing RawData
RawData = pd.read_csv('coffee_data.csv')

#first look into data
RawData.dtypes
for col in RawData.select_dtypes(['object']):
    print(RawData[col].value_counts())

RawData.describe()
RawData.skew()
RawData.hist(bins = 30)


#Visualization of data
sns.stripplot(data = RawData, y = 'coffee_amount', x = 'brewing_time', hue = 'mark')
sns.violinplot(data = RawData, x = 'region', y = 'mark', hue = 'preinfusion', split = True,
               palette = {0:'r', 1:'g'})
sns.violinplot(data = RawData, x = 'processing_method', y = 'mark', hue = 'preinfusion', split = True,
               palette = {0:'r', 1:'g'})
sns.boxplot(data = RawData, x = 'mark', y = 'brewing_temp')
sns.boxplot(data = RawData, x = 'mark', y = 'coffee_amount')
sns.violinplot(data = RawData, x = 'mark', y = 'water_ph')
sns.violinplot(data = RawData, x = 'mark', y = 'brewing_time')
sns.violinplot(data = RawData, x = 'mark', y = 'plantation_height')


#Preparing the data
#Searching for NaN values
Nulls = pd.DataFrame(RawData.isnull().any(), columns = ['Nulls'])
Nulls['NumberOfNan'] = pd.DataFrame(RawData.isnull().sum())
Nulls['Percentage'] = round((RawData.isnull().mean()*100), 2)
print(Nulls)

#Creating levels of temp and time
RawData['temp_level'] = RawData['brewing_temp']
RawData.loc[RawData['temp_level'] < 89, ['temp_level']] = 1
RawData.loc[95 <= RawData['temp_level'], ['temp_level']] = 3
RawData.loc[89 <= RawData['temp_level'], ['temp_level']] = 2


RawData['time_level'] = RawData['brewing_time']
RawData.loc[RawData['time_level'] < 120, ['time_level']] = 1
RawData.loc[180 <= RawData['time_level'], ['time_level']] = 3
RawData.loc[120 <= RawData['time_level'], ['time_level']] = 2


#Filling NaN values
#Coffee Amount
GroupedCA = RawData.groupby(['grinding_level', 'time_level', 'temp_level'])
GroupedCA_median = GroupedCA.median()
GroupedCA_median = GroupedCA_median.reset_index()[['grinding_level', 'time_level', 'temp_level', 'coffee_amount']]

def CoffeeAmount(row):
    condition=((GroupedCA_median['grinding_level'] == row['grinding_level']) &
               (GroupedCA_median['time_level'] == row['time_level']) &
               (GroupedCA_median['temp_level'] == row['temp_level']))
    return GroupedCA_median[condition]['coffee_amount'].values[0]

def CAProcessing():
    global RawData
    RawData['coffee_amount'] = RawData.apply(lambda row: CoffeeAmount(row)
    if np.isnan(row['coffee_amount']) else row['coffee_amount'], axis = 1)
    return RawData
RawData = CAProcessing()

RawData.coffee_amount.fillna(value = RawData['coffee_amount'].mean(), inplace = True)

#TDS
GroupedTDS = RawData.groupby(['temp_level', 'time_level', 'grinding_level'])
GroupedTDS_median = GroupedTDS.median()
GroupedTDS_median = GroupedTDS_median.reset_index()[['temp_level', 'time_level', 'grinding_level', 'TDS']]

def TDS(row):
    condition = ((GroupedTDS_median['temp_level'] == row['temp_level']) &
                 (GroupedTDS_median['time_level'] == row['time_level']) &
                 (GroupedTDS_median['grinding_level'] == row['grinding_level']))
    return GroupedTDS_median[condition]['TDS'].values[0]

def TDSProcessing():
    global RawData
    RawData['TDS'] = RawData.apply(lambda row: TDS(row)
    if np.isnan(row['TDS']) else row['TDS'], axis = 1)
    return RawData
RawData = TDSProcessing()

RawData.TDS.fillna(value = RawData['TDS'].mean(), inplace = True)

#Preinfusion
GroupedPre = RawData.groupby([ 'grinding_level', 'processing_method'])
GroupedPre_median = GroupedPre.median()
GroupedPre_median = GroupedPre_median.reset_index()[['grinding_level', 'processing_method', 'preinfusion']]

def Pre(row):
    condition = ((GroupedPre_median['grinding_level'] == row['grinding_level']) &
                 (GroupedPre_median['processing_method'] == row['processing_method']))
    return GroupedPre_median[condition]['preinfusion'].values[0]

def PreProcessing():
    global RawData
    RawData['preinfusion'] = RawData.apply(lambda row: Pre(row)
    if np.isnan(row['preinfusion']) else row['preinfusion'], axis = 1)
    return RawData
RawData = PreProcessing()


#Adding and remving columns
RawData['energy'] = RawData['brewing_time'] * RawData['brewing_temp']
RawData['robustness'] = RawData['grinding_level'] / RawData['TDS']
RawData['essential'] = RawData['coffee_amount'] * RawData['brewing_time']
RawData['barist_rank'] = RawData['region'].map({'Kenya':12, 'Panama':1, 'Rwanda':1, 'Colombia':10,
                                                'Ethiopia':25, 'Honduras':3, 'Brazil':0, 'Guatemala':7})
RawData['humidity'] = RawData['processing_method'].map({'Natural':0, 'Honey':2, 'Pulped Natural':1, 'Washed':5})
RawData['fermentation'] = RawData['processing_method'].map({'Natural':5, 'Honey':2, 'Pulped Natural':1, 'Washed':3})

RawData.drop(['temp_level', 'time_level'], axis = 1, inplace = True)

#Splitting for dependent and independent values
cols = [col for col in RawData.columns if col != 'mark']
X = RawData[cols]
Y = RawData.iloc[:, 10]


#Getting dummies
#region
def RegionDummies():
    global X
    region_dummies = pd.get_dummies(X['region'], prefix = 'region')
    X = pd.concat([X, region_dummies], axis = 1)
    X.drop('region', axis = 1, inplace = True)
    return X
X = RegionDummies()

def ProcessDummies():
    global X
    process_dummies = pd.get_dummies(X['processing_method'], prefix = 'processing_method')
    X = pd.concat([X, process_dummies], axis = 1)
    X.drop('processing_method', axis = 1, inplace = True)
    return X
X = ProcessDummies()

#Scalling the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 5))
X = scaler.fit_transform(X)
X = pd.DataFrame(X)

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

#Looking for most important values
from sklearn.feature_selection import RFE
acc_all =[]
stab_all = []
for t in np.arange(0, 20, 1):
    acc_loop = []
    stab_loop = []
    for b in np.arange(3, 13, 1):
        selector = RFE(XGB, b, 1)
        cv = cross_val_score(XGB, X_train.iloc[:, selector.fit(X, Y).support_],
                             Y_train, cv = 10, scoring = 'accuracy')
        acc_loop.append(cv.mean())
        stab_loop.append(cv.std()*100/cv.mean())
    acc_all.append(acc_loop)
    stab_all.append(stab_loop)
acc = pd.DataFrame(acc_all, columns = np.arange(3 , 13, 1))
stab = pd.DataFrame(stab_all, columns = np.arange(3, 13, 1))
print(acc.agg('mean'))
print(stab.agg('mean'))

selector = RFE(XGB, 9, 1)
values = X_train.iloc[:, selector.fit(X_train, Y_train).support_].columns
print(values)

XGB.fit(X_train, Y_train)
features = pd.DataFrame()
features['feature'] = X_train.columns
features['importance'] = XGB.feature_importances_
features.sort_values(by = 'importance', ascending = True, inplace = True)
features.set_index('feature', inplace = True)
features.plot(kind = 'barh')


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