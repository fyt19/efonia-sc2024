import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

import os
fileList=list()
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        fileList.append(os.path.join(dirname, filename))

for i in fileList: print(i) 

data = pd.read_csv(fileList[1], on_bad_lines='skip', nrows=99999999, low_memory=False)
data.columns = data.iloc[2]
data = data[3:]
data = data.drop(['table','_start','_stop'], axis=1) 

result = data.axes

print(result)

data["_time"] = pd.to_datetime(data['_time'])
data["_value"] = pd.to_numeric(data['_value'])
data['_field'] = data['_field'].astype(str)
data['_measurement'] = data['_measurement'].astype(str)

print(data.dtypes)

data.set_index('_time', inplace=False)

data=data[['_time','_value','_field','_measurement']]
data.head()

data.groupby('_field')['_value'].describe()

for field in data['_field'].unique():
    sns.histplot(data[data['_field'] == field]['_value'], kde=True)
    plt.title(f'Histogram of {field}')
    plt.show()

