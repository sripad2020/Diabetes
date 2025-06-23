import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, matthews_corrcoef, cohen_kappa_score,
                             confusion_matrix, roc_curve, precision_recall_curve, jaccard_score)
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from lightgbm import LGBMClassifier
import joblib


data = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')
print("Columns:", data.columns)
print("\nData Info:")
print(data.info())
print("\nMissing Values:")
print(data.isna().sum())
print("\nData Description:")
print(data.describe())
print("\nValue Counts per Column:")
for i in data.columns:
    print(f"\n{i}:")
    print(data[i].value_counts())

print("\nOriginal Data Length:", len(data))

sns.boxplot(x='Diabetes_012', y='BMI', data=data)
plt.title('BMI Distribution by Diabetes Class')
plt.show()

thresh = 3
for i in data.columns:
    data['z_Scores'] = (data[i] - data[i].mean()) / data[i].std()
    outliers = np.abs(data['z_Scores'] > thresh).sum()
    if outliers > 3:
        upper = data[i].mean() + thresh * data[i].std()
        lower = data[i].mean() - thresh * data[i].std()
        data = data[(data[i] > lower) & (data[i] < upper)]
    data = data.drop(columns=['z_Scores'])

print("Data Length after Outlier Removal:", len(data))

sns.boxplot(x='Diabetes_012', y='BMI', data=data)
plt.title('BMI Distribution by Diabetes Class')
plt.show()


# Feature selection based on correlation
corr = data.corr()['Diabetes_012']
corr = corr.drop(['Diabetes_012'])
x = [i for i in corr.index if corr[i] > 0]
x = data[x]
y = data['Diabetes_012']

smote = SMOTE(random_state=30)
x, y = smote.fit_resample(x, y)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)


models = {
    'Random_forest_classifier': RandomForestClassifier(max_depth=7, random_state=42),
    'Extra_Tree_Classifier': ExtraTreesClassifier(n_estimators=150, random_state=42),
    'XGB_classifier': XGBClassifier(random_state=42),
    'LGB_classifier': LGBMClassifier(n_estimators=150, random_state=42),
    'XGB_RF_Classifier': XGBRFClassifier(random_state=42),
    'Gradient_Boosting': GradientBoostingClassifier(random_state=42),
    'ada_boost_classifier': AdaBoostClassifier(random_state=42)
}


fpr_dict, tpr_dict, roc_auc_dict = {}, {}, {}
prec_dict, rec_dict, pr_auc_dict = {}, {}, {}
metrics_results = {name: {} for name in models.keys()}

for name, model in models.items():
    model.fit(x_train, y_train)
    print(f"{name} Test Score: {model.score(x_test, y_test)}")

joblib.dump(models['XGB_classifier'], 'diabetes_012_XGB.pkl')
joblib.dump(models['Extra_Tree_Classifier'], 'diabetes_012_ExtraTrees.pkl')
