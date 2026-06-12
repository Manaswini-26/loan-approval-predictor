import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

df = pd.read_csv('loan_data.csv')
X = df.drop('Loan_Status', axis=1)

model = joblib.load('best_model.pkl')
preprocessor = model.named_steps['preprocessor']
classifier = model.named_steps['classifier']

X_transformed = preprocessor.transform(X.sample(300, random_state=42))

feature_names = (
    ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
     'Credit_History', 'Dependents'] +
    ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
)

explainer = shap.TreeExplainer(classifier)
shap_values = explainer.shap_values(X_transformed)

plt.figure()
shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=100, bbox_inches='tight')
print("Saved shap_summary.png")
