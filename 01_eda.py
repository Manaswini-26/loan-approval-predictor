import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('loan_data.csv')

print("=== Shape ===")
print(df.shape)

print("\n=== Missing Values ===")
print(df.isnull().sum())

print("\n=== Target Distribution ===")
print(df['Loan_Status'].value_counts(normalize=True))

# --- Visualizations ---
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

sns.countplot(data=df, x='Loan_Status', ax=axes[0,0])
axes[0,0].set_title('Loan Status Distribution')

sns.histplot(df['ApplicantIncome'], kde=True, ax=axes[0,1])
axes[0,1].set_title('Applicant Income Distribution')

sns.histplot(df['LoanAmount'], kde=True, ax=axes[0,2])
axes[0,2].set_title('Loan Amount Distribution')

sns.countplot(data=df, x='Credit_History', hue='Loan_Status', ax=axes[1,0])
axes[1,0].set_title('Credit History vs Loan Status')

sns.countplot(data=df, x='Property_Area', hue='Loan_Status', ax=axes[1,1])
axes[1,1].set_title('Property Area vs Loan Status')

sns.boxplot(data=df, x='Loan_Status', y='ApplicantIncome', ax=axes[1,2])
axes[1,2].set_title('Income by Loan Status')

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=100)
print("\nSaved eda_plots.png")

# Correlation heatmap (numeric only)
plt.figure(figsize=(8,6))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=100)
print("Saved correlation_heatmap.png")
