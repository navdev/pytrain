import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

%matplotlib inline

df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

df.sample(5)

df.drop('customerID', axis='columns', inplace=True)

df.dtypes

df.TotalCharges.values

df.MonthlyCharges.values

#df.TotalCharges = pd.to_numeric(df.TotalCharges)

df[pd.to_numeric(df.TotalCharges, errors='coerce').isnull()]

df = df[df.TotalCharges != ' ']
df.shape

df.TotalCharges = pd.to_numeric(df.TotalCharges)

df.dtypes

tenure_churn_no = df[df.Churn=='No'].tenure
tenure_churn_yes = df[df.Churn=='Yes'].tenure
plt.xlabel('Tenure')
plt.ylabel('No of Customers')
plt.title('Customer Churn Visualization')
plt.hist([tenure_churn_yes, tenure_churn_no], color=['green', 'red'], label=['Churn=Yes', 'Churn=No'])
plt.legend()

mc_churn_no = df[df.Churn=='No'].MonthlyCharges
mc_churn_yes = df[df.Churn=='Yes'].MonthlyCharges
plt.xlabel('Monthly Charges')
plt.ylabel('No of Customers')
plt.title('Customer Churn Visualization (Monthly Charges)')
plt.hist([mc_churn_yes, mc_churn_no], color=['green', 'red'], label=['Churn=Yes', 'Churn=No'])
plt.legend()

