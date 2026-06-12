import numpy as np
import pandas as pd

np.random.seed(42)
n = 2000

data = pd.DataFrame({
    'Gender': np.random.choice(['Male', 'Female'], n, p=[0.65, 0.35]),
    'Married': np.random.choice(['Yes', 'No'], n, p=[0.6, 0.4]),
    'Dependents': np.random.choice([0, 1, 2, 3], n, p=[0.5, 0.2, 0.2, 0.1]),
    'Education': np.random.choice(['Graduate', 'Not Graduate'], n, p=[0.75, 0.25]),
    'Self_Employed': np.random.choice(['Yes', 'No'], n, p=[0.15, 0.85]),
    'ApplicantIncome': np.random.gamma(5, 1200, n).astype(int),
    'CoapplicantIncome': np.random.gamma(2, 800, n).astype(int),
    'LoanAmount': np.random.gamma(4, 35, n).astype(int),
    'Loan_Amount_Term': np.random.choice([120, 180, 240, 300, 360], n, p=[0.05, 0.1, 0.1, 0.15, 0.6]),
    'Credit_History': np.random.choice([1, 0], n, p=[0.84, 0.16]),
    'Property_Area': np.random.choice(['Urban', 'Semiurban', 'Rural'], n, p=[0.4, 0.35, 0.25]),
})

# introduce some missing values
for col in ['Gender', 'Dependents', 'Self_Employed', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']:
    mask = np.random.rand(n) < 0.04
    data.loc[mask, col] = np.nan

# create target with logical relationships + noise
score = (
    0.00008 * data['ApplicantIncome'].fillna(0)
    + 0.00005 * data['CoapplicantIncome'].fillna(0)
    - 0.01 * data['LoanAmount'].fillna(data['LoanAmount'].median())
    + 1.8 * data['Credit_History'].fillna(0)
    + 0.3 * (data['Education'] == 'Graduate')
    + 0.2 * (data['Married'] == 'Yes')
    + np.random.normal(0, 1, n)
)

threshold = np.percentile(score, 32)  # ~32% default rate
data['Loan_Status'] = np.where(score > threshold, 'Y', 'N')

data.to_csv('loan_data.csv', index=False)
print(data.shape)
print(data['Loan_Status'].value_counts())
print(data.isnull().sum())
