import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE,RandomOverSampler
from scipy import stats
from collections import Counter
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,roc_auc_score, average_precision_score, matthews_corrcoef, cohen_kappa_score,confusion_matrix, roc_curve, precision_recall_curve, jaccard_score
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from lightgbm import LGBMClassifier
import joblib

# Load data
data = pd.read_csv('diabetes_binary_5050split_health_indicators_BRFSS2015.csv')

# Data exploration
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

# Outlier removal using z-scores
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

# Feature selection based on correlation
corr = data.corr()['Diabetes_binary']
corr = corr.drop(['Diabetes_binary'])
x = [i for i in corr.index if corr[i] > 0]
x = data[x]
y = data['Diabetes_binary']

print(x.columns)
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

# Apply SMOTE for oversampling
smote = SMOTE(random_state=42)
x, y = smote.fit_resample(x, y)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)


models = {
    'Random_forest_classifier': RandomForestClassifier(max_depth=7, random_state=42),
    'Extra_Tree_Classifier': ExtraTreesClassifier(n_estimators=150, random_state=42),
    'XGB_classifier': XGBClassifier(random_state=42),
    'LGB_classifier': LGBMClassifier(n_estimators=150, random_state=42),
    'XGB_RF_Classifier': XGBRFClassifier(random_state=42),
    'Gradient_Boosting': GradientBoostingClassifier(random_state=42),
    'ada_boost_classifier': AdaBoostClassifier(random_state=42)
}

# Dictionaries for ROC and PR curves
fpr_dict, tpr_dict, roc_auc_dict = {}, {}, {}
prec_dict, rec_dict, pr_auc_dict = {}, {}, {}
metrics_results = {name: {} for name in models.keys()}

# Train models, evaluate, and generate visualizations
for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(x_test)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    metrics_results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Balanced Accuracy': balanced_accuracy_score(y_test, y_pred),
        'Precision (Weighted)': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'Recall (Weighted)': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'F1-Score (Weighted)': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'Precision (Macro)': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'Recall (Macro)': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'F1-Score (Macro)': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'ROC AUC': roc_auc_score(y_test, y_prob),
        'Average Precision': average_precision_score(y_test, y_prob),
        'MCC': matthews_corrcoef(y_test, y_pred),
        'Cohen’s Kappa': cohen_kappa_score(y_test, y_pred),
        'Specificity': specificity,
        'NPV': npv,
        'Jaccard Score': jaccard_score(y_test, y_pred)
    }

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    fpr_dict[name], tpr_dict[name] = fpr, tpr
    roc_auc_dict[name] = roc_auc_score(y_test, y_prob)
    prec_dict[name], rec_dict[name] = prec, rec
    pr_auc_dict[name] = average_precision_score(y_test, y_prob)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_{name}.png')
    plt.close()

# Save metrics to CSV
metrics_df = pd.DataFrame(metrics_results).T
metrics_df.to_csv('classification_metrics.csv')

# Plot ROC curves
plt.figure(figsize=(10, 8))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'pink'])
for name, color in zip(models.keys(), colors):
    plt.plot(fpr_dict[name], tpr_dict[name], label=f'{name} (AUC = {roc_auc_dict[name]:.2f})', color=color)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='lower right')
plt.savefig('roc_curves.png')
plt.close()

# Plot Precision-Recall curves
plt.figure(figsize=(10, 8))
for name, color in zip(models.keys(), colors):
    plt.plot(rec_dict[name], prec_dict[name], label=f'{name} (AP = {pr_auc_dict[name]:.2f})', color=color)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend(loc='lower left')
plt.savefig('pr_curves.png')
plt.close()

# Plot metrics comparison
metrics_df.plot(kind='bar', figsize=(15, 8))
plt.title('Model Comparison Across Metrics')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('metrics_comparison.png')
plt.close()

# Print classification metrics
print("\nClassification Metrics:")
print(metrics_df)

# Save XGBoost and Extra Trees models
joblib.dump(models['XGB_classifier'], 'diabetes_5050_XGB.pkl')
joblib.dump(models['Extra_Tree_Classifier'], 'diabetes_5050_ExtraTrees.pkl')
