import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE,RandomOverSampler
from collections import Counter
from scipy import stats
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


corr = data.corr()['Diabetes_012']
corr = corr.drop(['Diabetes_012'])
x = [i for i in corr.index if corr[i] > 0]
x = data[x]
y = data['Diabetes_012']

plt.figure(figsize=(6, 4))
sns.countplot(x=y)
plt.title("Original Class Distribution")
plt.show()

print("Original Distribution:", dict(Counter(y)))
smote = SMOTE(random_state=30)
x_smote, y_smote = smote.fit_resample(x, y)

plt.figure(figsize=(6, 4))
sns.countplot(x=y_smote)
plt.title("After SMOTE")
plt.show()

print("SMOTE Distribution:", dict(Counter(y_smote)))

# ---------- 4. Majority Oversampling (oversample majority class only) ----------
# This is rare, so we forcefully replicate majority class to equal 90%
ros_major = RandomOverSampler(random_state=30)
x_major, y_major = ros_major.fit_resample(x, y)

plt.figure(figsize=(6, 4))
sns.countplot(x=y_major)
plt.title("After Majority Oversampling")
plt.show()

print("Majority Oversampling:", dict(Counter(y_major)))

# ---------- 5. Minority Oversampling (oversample minority class only) ----------
ros_minor = RandomOverSampler(sampling_strategy='minority', random_state=30)
x_minor, y_minor = ros_minor.fit_resample(x, y)

plt.figure(figsize=(6, 4))
sns.countplot(x=y_minor)
plt.title("After Minority Oversampling")
plt.show()

print("Minority Oversampling:", dict(Counter(y_minor)))

# ---------- 6. Calculate Skewness, Std Dev, Kurtosis ----------
skewness = []
std_dev = []
kurtosis = []

for col in x.columns:
    skewness.append(x[col].skew())
    std_dev.append(x[col].std())
    kurtosis.append(x[col].kurtosis())

stats_df = pd.DataFrame({
    'Feature': x.columns,
    'Skewness': skewness,
    'Standard Deviation': std_dev,
    'Kurtosis': kurtosis
})

# ---------- 7. Graphical Representation of Statistics ----------
plt.figure(figsize=(15, 12))

plt.subplot(3, 1, 1)
sns.barplot(data=stats_df, x='Feature', y='Skewness')
plt.title('Skewness of Each Feature')
plt.xticks(rotation=90)

plt.subplot(3, 1, 2)
sns.barplot(data=stats_df, x='Feature', y='Standard Deviation')
plt.title('Standard Deviation of Each Feature')
plt.xticks(rotation=90)

plt.subplot(3, 1, 3)
sns.barplot(data=stats_df, x='Feature', y='Kurtosis')
plt.title('Kurtosis of Each Feature')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

# ---------- 8. 5 Distribution Plots for Top 3 Features ----------
top_features = x.columns[:3]

for feature in top_features:
    plt.figure(figsize=(18, 10))

    plt.subplot(2, 3, 1)
    sns.histplot(x[feature], kde=True, bins=30)
    plt.title(f'Histogram of {feature}')

    plt.subplot(2, 3, 2)
    sns.boxplot(x=x[feature])
    plt.title(f'Boxplot of {feature}')

    plt.subplot(2, 3, 3)
    sns.violinplot(x=x[feature])
    plt.title(f'Violin Plot of {feature}')

    plt.subplot(2, 3, 4)
    sorted_data = np.sort(x[feature])
    y_ecdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
    plt.plot(sorted_data, y_ecdf, marker='.', linestyle='none')
    plt.title(f'ECDF of {feature}')
    plt.xlabel(feature)
    plt.ylabel('ECDF')

    plt.subplot(2, 3, 5)
    stats.probplot(x[feature], dist="norm", plot=plt)
    plt.title(f'Q-Q Plot of {feature}')

    plt.tight_layout()
    plt.show()

# ---------- 9. Print Summary Statistics ----------
print("\n========== FEATURE STATISTICS ==========")
for index, row in stats_df.iterrows():
    print(f"Feature: {row['Feature']}")
    print(f"  Skewness           : {row['Skewness']:.4f}")
    print(f"  Standard Deviation : {row['Standard Deviation']:.4f}")
    print(f"  Kurtosis           : {row['Kurtosis']:.4f}")
    print("--------------------------------------------------")

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
