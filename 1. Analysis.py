import pandas as pd
import seaborn as sns
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