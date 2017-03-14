
# coding: utf-8

# In[300]:

get_ipython().magic(u'matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib as plt

df = pd.read_csv("/Users/arunash/kaggle-titanic/loan_prediction/data/train.csv") #Reading the dataset in a dataframe using Pandas

test_data = pd.read_csv("/Users/arunash/kaggle-titanic/loan_prediction/data/test.csv") #Reading the dataset in a dataframe


# In[301]:

df.head(10) #Print first 10 rows


# In[302]:

df.describe() #get numerical summary
#Some inferences can be drawn from this
#LoanAmount has 614-592 = 22 missing values.
#Loan_Amount_Term has (614 – 600) 14 missing values.
#Credit_History has (614 – 564) 50 missing values.
#84% applicants have a credit_history. How? The mean of Credit_History field is 0.84 (Remember, Credit_History has value 1 for those who have a credit history and 0 otherwise)
#ApplicantIncome distribution seems to be in line with expectation. Same with CoapplicantIncome
#We can get idea of a possible skew in the data by comparing the mean to the median, i.e. the 50% figure.


# In[303]:

df['Property_Area'].value_counts()


# In[304]:

df['Credit_History'].value_counts()


# In[305]:

#Distribution Analysis
get_ipython().magic(u'matplotlib inline')
df['ApplicantIncome'].hist(bins=50)


# In[306]:

#Distribution Analysis
get_ipython().magic(u'matplotlib inline')
df['LoanAmount'].hist(bins=50)


# In[307]:

df.boxplot(column='ApplicantIncome')
#outliers and extreme values due to income disparities different factors contributing to it like education


# In[308]:

df.boxplot(column='ApplicantIncome', by = 'Education')
#no substantial different between the 
#mean income of graduate and non-graduates. 
#But there are a higher number of graduates with very high incomes, 
#which are appearing to be the outliers.



# In[309]:

df.boxplot(column='LoanAmount')
#both ApplicantIncome and LoanAmount require some amount of data munging.
#LoanAmount has missing and well as extreme values values, 
#while ApplicantIncome has a few extreme values, 
#which demand deeper understanding. #


# In[310]:

#1 – Boolean Indexing
# filter values of a column based on conditions from another set of columns
#all females who are not graduate and got a loan
df.loc[(df["Gender"]=="Female") & (df["Education"]=="Not Graduate") & (df["Loan_Status"]=="Y"), ["Gender","Education","Loan_Status"]]


# In[311]:

#Apply Function
#Create a new function:
def num_missing(x):
  return sum(x.isnull())

#Applying per column:
print "Missing values per column:"
print df.apply(num_missing, axis=0) #axis=0 defines that function is to be applied on each column

#Applying per row:
print "\nMissing values per row:"
print df.apply(num_missing, axis=1).head() #axis=1 defines that function is to be applied on each row


# In[312]:

#Pivot table
#key column is “LoanAmount” which has missing values. 
#We can impute it using mean amount of 
#each ‘Gender’, ‘Married’ and ‘Self_Employed’ group. 
#The mean ‘LoanAmount’ of each group can be determined as:

impute_grps = df.pivot_table(values=["LoanAmount"], index=["Gender","Married","Self_Employed"], aggfunc=np.mean)
print impute_grps


# In[313]:

#Pivot table on credit history
temp1 = df['Credit_History'].value_counts(ascending=True)
temp2 = df.pivot_table(values='Loan_Status',index=['Credit_History'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())
print 'Frequency Table for Credit History:' 
print temp1

print '\nProbility of getting loan for each Credit History class:' 
print temp2


# In[314]:

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('Count of Applicants')
ax1.set_title("Applicants by Credit_History")
temp1.plot(kind='bar')

ax2 = fig.add_subplot(122)
temp2.plot(kind = 'bar')
ax2.set_xlabel('Credit_History')
ax2.set_ylabel('Probability of getting loan')
ax2.set_title("Probability of getting loan by credit history")

#the chances of getting a loan are eight-fold if the applicant has a valid credit history.
#You can plot similar graphs by Married, Self-Employed, Property_Area, etc.


# In[315]:

temp3 = pd.crosstab(df['Credit_History'], df['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
temp4 = pd.crosstab([df['Gender'], df['Credit_History']], df['Loan_Status'])
temp4.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)

# two basic classification algorithms here,
#one based on credit history, while other on 2 categorical variables 
#(including gender)


# In[316]:

#Data munging 
#fixing problems in the data set, which needs to be solved before the data 
#is ready for a good model. This exercise is typically referred as “Data Munging”.
#Missing values + Extreme values
#In addition to these problems with numerical fields,
#we should also look at the non-numerical fields 
#i.e. Gender, Property_Area, Married, Education and Dependents

#Finding missing values:
df.apply(lambda x: sum(x.isnull()),axis=0) 


# In[317]:

#Filling missing amounts in loan amoutn
#df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean(), inplace=True)
#Other way is build a supervised learning model to predict loan amount 
# on the basis of other variables and then use age along with other variables to predict survival.


# In[318]:

#Lets take a hypothesis is that the whether a 
#person is educated or self-employed can combine to 
#give a good estimate of loan amount.
df.boxplot(column='LoanAmount', by = ['Education', 'Self_Employed'])


# In[319]:

#Self_Employed has missing values 
df['Self_Employed'].value_counts()


# In[320]:

#Majority is no so impute Self_Employed to No
df['Self_Employed'].fillna('No',inplace=True)


# In[321]:

table = df.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)
# Define function to return value of this pivot_table
def fage(x):
 return table.loc[x['Self_Employed'],x['Education']]
# Replace missing values
df['LoanAmount'].fillna(df[df['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)


# In[322]:

#treat for extreme values in distribution of LoanAmount and ApplicantIncome
# Let’s analyze LoanAmount first. Since the extreme values 
#are practically possible, i.e. some people might apply for 
#high value loans due to specific needs. So instead of 
#treating them as outliers, 
#let’s try a log transformation to nullify their effect:
df['LoanAmount_log'] = np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins=20)

#distribution looks much closer to normal and 
#effect of extreme values has been significantly subsided.


# In[323]:

#Coming to ApplicantIncome.
#One intuition can be that some applicants have lower income 
#but strong support Co-applicants. 
#So it might be a good idea to combine both incomes 
#as total income and take a log transformation of the same.

df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome_log'] = np.log(df['TotalIncome'])
df['LoanAmount_log'].hist(bins=20) 



# In[324]:

#Building predictive model
# Skicit-Learn (sklearn) is the most commonly used library 
#in Python for this purpose
# sklearn requires all inputs to be numeric, 
#we should convert all our categorical variables into numeric
#by encoding the categories. 
#This can be done using the following code:
from sklearn.preprocessing import LabelEncoder
var_mod = ['Credit_History','Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
var_mod1 = ['Credit_History','Gender','Married','Dependents','Education','Self_Employed','Property_Area']
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i])
df.dtypes 
for i in var_mod1:
    test_data[i] = le.fit_transform(df[i])


# In[ ]:

#Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

#Generic function for making a classification model and accessing performance:
def classification_model(model, data, predictors, outcome):
  #Fit the model:
  model.fit(data[predictors],data[outcome])
  
  #Make predictions on training set:
  predictions = model.predict(data[predictors])
  
  #Print accuracy
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print "Accuracy : %s" % "{0:.3%}".format(accuracy)

  #Perform k-fold cross-validation with 5 folds
  kf = KFold(data.shape[0], n_folds=5)
  error = []
  for train, test in kf:
    # Filter training data
    train_predictors = (data[predictors].iloc[train,:])
    
    # The target we're using to train the algorithm.
    train_target = data[outcome].iloc[train]
    
    # Training the algorithm using the predictors and target.
    model.fit(train_predictors, train_target)
    
    #Record error from each cross-validation run
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
 
  print "Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error))

  #Fit the model again so that it can be refered outside the function:
  model.fit(data[predictors],data[outcome])


# In[ ]:




# In[ ]:

#So let’s make our first model with ‘Credit_History’.
#We can try different combination of variables:
outcome_var = 'Loan_Status'
model = LogisticRegression()
predictor_var = ['Credit_History','Education','Married','Self_Employed']
classification_model(model, df,predictor_var,outcome_var)
print model.predict(test_data[predictor_var])


# In[ ]:

model = DecisionTreeClassifier()
predictor_var = ['Credit_History','Gender','Married','Education']
classification_model(model, df,predictor_var,outcome_var)


# In[ ]:

predictor_var = ['Credit_History','Loan_Amount_Term','LoanAmount_log']
classification_model(model, df,predictor_var,outcome_var)


# In[ ]:

model = RandomForestClassifier(n_estimators=100)
predictor_var = ['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'Property_Area',
        'LoanAmount_log','TotalIncome_log']
classification_model(model, df,predictor_var,outcome_var)


# In[ ]:




# In[ ]:



