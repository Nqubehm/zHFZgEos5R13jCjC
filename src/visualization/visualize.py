# Imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from dython import nominal

plt.figure(figsize=(8,8))
plt.title('Counts in Different Job Types as Per Marital Status')
sns.countplot(y = 'job', data=data, hue='marital', order=data.job.value_counts().index);

plt.title('Yearly Bank Balances shown for Different Age Groups as Per Term Deposit Subscription')
sns.scatterplot(x='age', y='balance', hue='y', data=data);

plt.figure(figsize=(8,8))
plt.title('Yearly Bank Balances shown For different Job Types as Per Marital Status')
sns.barplot(y='job', x='balance', hue='marital', data=data);

plt.title('Job Types as Per Term Deposit Subscription')
sns.countplot(y=data.job, hue=data.y, 
              order=data.job.value_counts().index, data=data);

plt.title('Contact Communication Types as Per Term Deposit Subscription')
sns.countplot(y=data.contact, hue=data.y, 
              order=data.contact.value_counts().index, data=data);
              
plt.title('Term Deposit Subscriptions in Relation To Campaigns Performed')
sns.barplot(x='y', y='campaign', data=data);   

plt.title('How Likely Are People with Housing Loan to Take a Term Deposit as Per Age Group?')
sns.barplot(x='housing', y='age', hue='y',
           data=data);

plt.title('How Likely Are People with Personal Loan to Take a Term Deposit as Per Age Group?')
sns.barplot(x='loan', y='age', hue='y',
           data=data);
           
plt.title('Personal Loan Vs Yearly Bank Balance as Per Deposit Subscription?')
sns.barplot(x='loan', y='balance', hue='y',
           data=data);
 
plt.title('Housing Loan vs Yearly Bank Balance as Per Term Deposit Subscription')
sns.barplot(x='housing', y='balance', hue='y',
           data=data);
           
# Correlation Plot
nominal.associations(data,figsize=(20,10),mark_columns=True);    

plt.title('Term Deposit Subscription')
sns.countplot(x=data.y);

plt.title('Term Deposit Subscription - SMOTE Applied')
sns.countplot(x=y_train_smote);
           
           
