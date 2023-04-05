# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 16:56:27 2023

@author: Mriganka

@client: Upgrad

@Topic: Lead Score - Case Study
"""

# Supress Warnings
import warnings
warnings.filterwarnings("ignore")

# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, confusion_matrix 
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import precision_recall_curve

# STEP 1: Import and inspect data
df = pd.read_csv("Leads.csv")
print(df.shape)
df.head()

## Removing the duplicates rows(if any)
df.drop_duplicates()
df.shape

df.info()
df.describe()

# STEP 2: Cleaning the data
## Replacing 'Select' with NaN
df = df.replace('Select',np.NaN)

## finding missing values
round(100*(df.isnull().sum()/len(df.index)),2)

## Dropping any coloums wirh more than 25% null value
df1 = df.drop(columns = ['Country','What is your current occupation','What matters most to you in choosing a course',
                         'Tags','Lead Quality','Lead Profile','Asymmetrique Activity Index','Asymmetrique Profile Index',
                         'Asymmetrique Activity Score','Asymmetrique Profile Score','Specialization',
                         'How did you hear about X Education','City'])
df1.shape

round(100*(df1.isnull().sum()/len(df1.index)),2)

## Replacing the rest blank with 'nan'
df1['Lead Source'] = df1['Lead Source'].fillna('NaN')
df1['TotalVisits'] = df1['TotalVisits'].fillna('NaN')
df1['Page Views Per Visit'] = df1['Page Views Per Visit'].fillna('NaN')
df1['Last Activity'] = df1['Last Activity'].fillna('NaN')

round(100*(df1.isnull().sum()/len(df1.index)),2)


## finding the effect of the "nan" in the data 
for column in df1:
    print(df1[column].astype('category').value_counts())
    
## There is no category with high NaN value, thus not Dropping the column with NaN

## Checking the column with a unique value as it will not effect the analysis
df1.nunique()

## Dropping the coulm with only single value
df2 = df1.drop(columns = ['Magazine','Receive More Updates About Our Courses',
                          'Update me on Supply Chain Content',
                          'Get updates on DM Content',
                          'I agree to pay the amount through cheque'])

## Removing 'Prospect ID' & 'Lead Number' values since this are unique.
df_final = df2.drop(columns = ['Prospect ID', 'Lead Number'])
df_final.shape

# STEP 3: Numeric Transformations of categorical variable & Get Dummies
## Converting Yes to 1 and No to 0
df_final['Do Not Email'] = df_final['Do Not Email'].map({'Yes':1, 'No':0})
df_final['Do Not Call'] = df_final['Do Not Call'].map({'Yes':1, 'No':0})
df_final['Search'] = df_final['Search'].map({'Yes':1, 'No':0})
df_final['X Education Forums'] = df_final['X Education Forums'].map({'Yes':1, 'No':0})
df_final['Newspaper'] = df_final['Newspaper'].map({'Yes':1, 'No':0})
df_final['Digital Advertisement'] = df_final['Digital Advertisement'].map({'Yes':1, 'No':0})
df_final['Through Recommendations'] = df_final['Through Recommendations'].map({'Yes':1, 'No':0})
df_final['A free copy of Mastering The Interview'] = df_final['A free copy of Mastering The Interview'].map({'Yes':1, 'No':0})

## Checking for the changes
for column in df_final:
    print(df_final[column].astype('category').value_counts())

## Creation of Dummy Variables
df_final.info()

df_final.loc[:, df_final.dtypes == 'object'].columns

## Create dummy variables using the 'get_dummies'
dummy = pd.get_dummies(df_final[['Lead Origin', 'Lead Source', 'TotalVisits', 
                                'Page Views Per Visit','Last Activity', 'Newspaper Article', 
                                'Last Notable Activity']],drop_first=True)

## Adding the results to the master dataframe
df_f_dum = pd.concat([df_final, dummy], axis = 1)
df_f_dum

df_f_dum = df_f_dum.drop(['Lead Origin','Lead Source', 'Do Not Email', 'Do Not Call','Last Activity', 
                          'Search','Newspaper Article', 'X Education Forums', 
                          'Newspaper','Digital Advertisement', 'Through Recommendations',
                          'A free copy of Mastering The Interview', 'Last Notable Activity'], 1)
df_f_dum

# STEP 4: EDA
## Looking at correlation matrix
plt.figure(figsize = (20,10))        
sns.heatmap(df_f_dum.corr(),annot = True)
plt.show()

# STEP 5: Train - Test Split
X = df_f_dum.drop(['Converted'], 1)
X.head()

## Putting the target variable in y
y = df_f_dum['Converted']
y.head()

## Split the dataset into 70% and 30% for train and test respectively
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=10)

## Scale the three numeric features with Minmx Scaler
scaler = MinMaxScaler()
X_train[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']] = scaler.fit_transform(X_train[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']])
X_train.head()

## To check the correlation among varibles
plt.figure(figsize=(20,30))
sns.heatmap(X_train.corr())
plt.show()

# STEP 6: Model Building
logreg = LogisticRegression()

## Running RFE with 15 variables as output
rfe = RFE(logreg, n_features_to_select=15)
rfe = rfe.fit(X_train, y_train)

## Features that have been selected by RFE
list(zip(X_train.columns, rfe.support_, rfe.ranking_))

## Put all the columns selected by RFE in the variable 'col'
col = X_train.columns[rfe.support_]

## Selecting columns selected by RFE
X_train = X_train[col]

X_train_sm = sm.add_constant(X_train)
logm1 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
res = logm1.fit()
res.summary()

## Make a VIF dataframe for all the variables present
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

X_train.drop('Last Notable Activity_had a phone conversation', axis = 1, inplace = True)

## Refit the model with the new set of features
X_train_sm = sm.add_constant(X_train)
logm2 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()

## Make a VIF dataframe for all the variables present
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

X_train.drop('What is your current occupation_housewife', axis = 1, inplace = True)

## Refit the model with the new set of features
X_train_sm = sm.add_constant(X_train)
logm3 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
res = logm3.fit()
res.summary()

## Make a VIF dataframe for all the variables present
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

X_train.drop('What is your current occupation_other', axis = 1, inplace = True)

## Refit the model with the new set of features
X_train_sm = sm.add_constant(X_train)
logm4 = sm.GLM(y_train, X_train_sm, family = sm.families.Binomial())
res = logm4.fit()
res.summary()

## Make a VIF dataframe for all the variables present
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

# STEP 7: Prediction

## Predicting the probabilities on the train set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]

## Reshaping to an array
y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]

## Data frame with given convertion rate and probablity of predicted ones
y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Conversion_Prob':y_train_pred})
y_train_pred_final.head()

## Substituting 0 or 1 with the cut off as 0.5
y_train_pred_final['Predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()

# STEP 8: Model Evaluation

print ('Accuracy: ', accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted))
print ('F1 score: ', f1_score(y_train_pred_final.Converted, y_train_pred_final.Predicted))
print ('Recall: ', recall_score(y_train_pred_final.Converted, y_train_pred_final.Predicted))
print ('Precision: ', precision_score(y_train_pred_final.Converted, y_train_pred_final.Predicted))
print ('\n clasification report:\n', classification_report(y_train_pred_final.Converted, y_train_pred_final.Predicted))
print ('\n confussion matrix:\n',confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted))

# STEP 9: ROC Curve

## ROC function
def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None

fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob, drop_intermediate = False )

## Call the ROC function
draw_roc(y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob)

## Creating columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()

## Creating a dataframe to see the values of accuracy, sensitivity, and specificity at different values of probabiity cutoffs
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])

## Making confusing matrix to find values of sensitivity, accurace and specificity for each level of probablity
from sklearn.metrics import confusion_matrix
num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
cutoff_df

## Plotting it
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()

y_train_pred_final['final_predicted'] = y_train_pred_final.Conversion_Prob.map( lambda x: 1 if x > 0.35 else 0)
y_train_pred_final.head()

print ('Accuracy: ', accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted))
print ('F1 score: ', f1_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted))
print ('Recall: ', recall_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted))
print ('Precision: ', precision_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted))
print ('\n clasification report:\n', classification_report(y_train_pred_final.Converted, y_train_pred_final.final_predicted))
print ('\n confussion matrix:\n',confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted))

# STEP 10: Prediction on Test set

## Scaling numeric values
X_test[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']] = scaler.transform(X_test[['TotalVisits', 'Page Views Per Visit', 'Total Time Spent on Website']])

## Select the columns in X_train for X_test as well
X_test = X_test[col]

## Add a constant to X_test
X_test_sm = sm.add_constant(X_test[col])
X_test_sm
X_test_sm

## Storing prediction of test set in the variable 'y_test_pred'
y_test_pred = res.predict(X_test_sm)

## Coverting it to df
y_pred_df = pd.DataFrame(y_test_pred)

## Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)

## Remove index for both dataframes to append them side by side 
y_pred_df.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)

## Append y_test_df and y_pred_df
y_pred_final = pd.concat([y_test_df, y_pred_df],axis=1)

## Renaming column 
y_pred_final= y_pred_final.rename(columns = {0 : 'Conversion_Prob'})
y_pred_final.head()

## Making prediction using cut off 0.35
y_pred_final['final_predicted'] = y_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.35 else 0)
y_pred_final

print ('Accuracy: ', accuracy_score(y_pred_final['Converted'], y_pred_final.final_predicted))
print ('F1 score: ', f1_score(y_pred_final['Converted'], y_pred_final.final_predicted))
print ('Recall: ', recall_score(y_pred_final['Converted'], y_pred_final.final_predicted))
print ('Precision: ', precision_score(y_pred_final['Converted'], y_pred_final.final_predicted))
print ('\n clasification report:\n', classification_report(y_pred_final['Converted'], y_pred_final.final_predicted))
print ('\n confussion matrix:\n',confusion_matrix(y_pred_final['Converted'], y_pred_final.final_predicted))

# STEP 11: Precision and Recall Trade off

y_train_pred_final.Converted, y_train_pred_final.Predicted

p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob)

plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()

y_train_pred_final['final_predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.41 else 0)
y_train_pred_final.head()

print ('Accuracy: ', accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted))
print ('F1 score: ', f1_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted))
print ('Recall: ', recall_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted))
print ('Precision: ', precision_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted))
print ('\n clasification report:\n', classification_report(y_train_pred_final.Converted, y_train_pred_final.final_predicted))
print ('\n confussion matrix:\n',confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted))

# STEP 12: Prediction on Test set

## Storing prediction of test set in the variable 'y_test_pred'
y_test_pred = res.predict(X_test_sm)

## Coverting it to df
y_pred_df = pd.DataFrame(y_test_pred)

## Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)

## Remove index for both dataframes to append them side by side 
y_pred_df.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)

## Append y_test_df and y_pred_df
y_pred_final = pd.concat([y_test_df, y_pred_df],axis=1)

## Renaming column 
y_pred_final= y_pred_final.rename(columns = {0 : 'Conversion_Prob'})
y_pred_final.head()

## Making prediction using cut off 0.41
y_pred_final['final_predicted'] = y_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.41 else 0)
y_pred_final

print ('Accuracy: ', accuracy_score(y_pred_final['Converted'], y_pred_final.final_predicted))
print ('F1 score: ', f1_score(y_pred_final['Converted'], y_pred_final.final_predicted))
print ('Recall: ', recall_score(y_pred_final['Converted'], y_pred_final.final_predicted))
print ('Precision: ', precision_score(y_pred_final['Converted'], y_pred_final.final_predicted))
print ('\n clasification report:\n', classification_report(y_pred_final['Converted'], y_pred_final.final_predicted))
print ('\n confussion matrix:\n',confusion_matrix(y_pred_final['Converted'], y_pred_final.final_predicted))




