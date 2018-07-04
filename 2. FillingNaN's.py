import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

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
