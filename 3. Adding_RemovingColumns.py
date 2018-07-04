import warnings
warnings.filterwarnings('ignore')

#Adding and remving columns
RawData['energy'] = RawData['brewing_time'] * RawData['brewing_temp']
RawData['robustness'] = RawData['grinding_level'] / RawData['TDS']
RawData['essential'] = RawData['coffee_amount'] * RawData['brewing_time']
RawData['barist_rank'] = RawData['region'].map({'Kenya':12, 'Panama':1, 'Rwanda':1, 'Colombia':10,
                                                'Ethiopia':25, 'Honduras':3, 'Brazil':0, 'Guatemala':7})
RawData['humidity'] = RawData['processing_method'].map({'Natural':0, 'Honey':2, 'Pulped Natural':1, 'Washed':5})
RawData['fermentation'] = RawData['processing_method'].map({'Natural':5, 'Honey':2, 'Pulped Natural':1, 'Washed':3})

RawData.drop(['temp_level', 'time_level'], axis = 1, inplace = True)