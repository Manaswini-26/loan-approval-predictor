import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('loan_data.csv')

X = df.drop('Loan_Status', axis=1)
y = (df['Loan_Status'] == 'Y').astype(int)  # 1 = approved/good, 0 = default risk

numeric_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                     'Loan_Amount_Term', 'Credit_History', 'Dependents']
categorical_features = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing pipelines
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
])

results = {}

def evaluate(name, model, X_test_t, y_test):
    preds = model.predict(X_test_t)
    probs = model.predict_proba(X_test_t)[:, 1]
    metrics = {
        'Accuracy': accuracy_score(y_test, preds),
        'Precision': precision_score(y_test, preds),
        'Recall': recall_score(y_test, preds),
        'F1': f1_score(y_test, preds),
        'ROC-AUC': roc_auc_score(y_test, probs)
    }
    results[name] = metrics
    print(f"\n=== {name} ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    return preds

# --- Baseline: Logistic Regression ---
log_reg = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))
])
log_reg.fit(X_train, y_train)
preds_lr = evaluate('Logistic Regression (Baseline)', log_reg, X_test, y_test)

# --- Random Forest ---
rf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
])
rf.fit(X_train, y_train)
preds_rf = evaluate('Random Forest', rf, X_test, y_test)

# --- XGBoost with hyperparameter tuning ---
xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(eval_metric='logloss', random_state=42))
])

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [3, 5],
    'classifier__learning_rate': [0.05, 0.1],
    'classifier__scale_pos_weight': [1, 2]
}

grid_search = GridSearchCV(xgb_pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)
print(f"\nBest XGBoost params: {grid_search.best_params_}")

best_xgb = grid_search.best_estimator_
preds_xgb = evaluate('XGBoost (Tuned)', best_xgb, X_test, y_test)

# --- Results comparison table ---
results_df = pd.DataFrame(results).T
results_df.to_csv('model_comparison.csv')
print("\n=== Model Comparison ===")
print(results_df)

# --- Confusion matrices ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, preds) in zip(axes, [('Logistic Regression', preds_lr),
                                      ('Random Forest', preds_rf),
                                      ('XGBoost (Tuned)', preds_xgb)]):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Reject', 'Approve'], yticklabels=['Reject', 'Approve'])
    ax.set_title(name)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=100)
print("\nSaved confusion_matrices.png")

# Save best model (XGBoost)
joblib.dump(best_xgb, 'best_model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')
print("\nSaved best_model.pkl")
