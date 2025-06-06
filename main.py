import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBRFClassifier,XGBClassifier
from lightgbm import DaskLGBMClassifier,LGBMClassifier

data=pd.read_csv('diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
print(data.columns)
print(data.info())
print(data.isna().sum())
print(data.describe())

for i in data.columns.values:
    print(data[i].value_counts())

print(len(data))
thresh=3

for i in data.columns.values:
    data['z_Scores']=(data[i]-data[i].mean())/data[i].std()
    outliers=np.abs(data['z_Scores']>3).sum()
    if outliers > 3:
        upper=data[i].mean()+thresh*data[i].std()
        lower=data[i].mean()-thresh*data[i].std()
        data=data[(data[i]>lower)&(data[i]<upper)]

print(len(data))

corr=data.corr()['Diabetes_binary']
corr=corr.drop(['Diabetes_binary','z_Scores'])
x=[i for i in corr.index if corr[i]>0]
x=data[x]
y=data['Diabetes_binary']

x_train,x_test,y_train,y_test=train_test_split(x,y)

rf=RandomForestClassifier(class_weight='balanced_subsample')
rf.fit(x_train,y_train)
print(rf.score(x_test,y_test))