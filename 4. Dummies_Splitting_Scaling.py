import pandas as pd
import warnings
warnings.filterwarnings('ignore')

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