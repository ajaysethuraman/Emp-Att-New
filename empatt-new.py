#!/usr/bin/env python
# coding: utf-8

# <h2><u>Problem Statement: </u> </h2> <h3>"Employee Attrition": The task is to find through predictive modelling, why employees leave the company prematurely and the redundant factors that cause this Attrition. </h3>
# 
# <h2><u>Project by:</u></h2><h3> Ajay Sethuraman</h3> <br>
# <h2><u>Under the guidance of:</u></h2> <h3> Mr. Muthuraja Sivanantham </h3> <br>

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings


# In[2]:


df=pd.read_csv("HR_comma_sep_2.csv")


# In[3]:


df.head(5)


# In[4]:


df.describe()


# In[5]:


df.shape


# ## Exploratory Analysis on the data
# <br> Here we are doing some basic exploratory analysis on the data, bringing up information on the data and a look at the data in a visual manner. We are also looking at the missing values and fixing them.

# In[6]:


df.info()


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
pd.crosstab(df.Department,df.left).plot(kind='bar')
plt.title('Attrition for Department')
plt.xlabel('Department')
plt.ylabel('Frequency of Attrition')
plt.savefig('department_bar_chart')


# In[8]:


table=pd.crosstab(df.salary, df.left)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Salary Level vs Turnover')
plt.xlabel('Salary Level')
plt.ylabel('Proportion of Employees')
plt.savefig('salary_bar_chart')


# Dataset contains 14999 rows and 10 columns, each row has the details of an employee.  
# 2 variables are categorical, remaining columns are of int and float
# 
# ### Checking for any missing values

# In[9]:


df.isnull().sum()


# As seen above the data contains some missing values, so we will fill them by mean for the numerical values and use ffill for the non-numerical value. <br>
# Come up with a percentage for missing values

# In[10]:


m1 = df['satisfaction_level'].mean()
m2 = df['number_project'].mean()
m3 = df['average_montly_hours'].mean()


# In[11]:


df["satisfaction_level"].fillna(value = m1, inplace = True)
df["number_project"].fillna(value = m2, inplace = True)
df["average_montly_hours"].fillna(value = m3, inplace = True)
df["Department"].fillna(method = 'ffill', inplace = True)


# In[12]:


df.isnull().sum()


# All the missing values have been filled.

# In[13]:


df


# In[14]:


## Let's separate numerical and categorical vaiables into 2 dfs

def sep_data(data):
    
    numerics = ['int32','float32','int64','float64']
    num_data = df.select_dtypes(include=numerics)
    cat_data = df.select_dtypes(exclude=numerics)
    
    return num_data, cat_data

num_data,cat_data = sep_data(df)


# In[15]:


df.Department.value_counts()


# In[16]:


df.salary.value_counts()


# <h3> Running VIF to check highly correlated values, since they may be redundant. </h3>

# In[17]:


features = num_data.drop(columns='left')
feature_list = "+".join(features.columns)
y, X = dmatrices('left~'+feature_list,num_data,return_type='dataframe')


# In[18]:


vif = pd.DataFrame()
vif['VIF Factor'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['Features'] = X.columns
vif


# The above table shows that there is no variable with a 'high" Variance Inflation Factor.
# So, this method suggests we should not drop any variable

# In[19]:


num_data.groupby('left').mean()


# In[20]:


fig,ax = plt.subplots(2,3, figsize=(17,12))               
sns.distplot(x= df['satisfaction_level'], ax = ax[0,0]) 
sns.distplot(df['last_evaluation'], ax = ax[0,1]) 
sns.countplot(df['number_project'], ax = ax[0,2]) 
sns.distplot(df['average_montly_hours'], ax = ax[1,0]) 
sns.distplot(df['time_spend_company'], ax = ax[1,1]) 
sns.countplot(df['promotion_last_5years'], ax = ax[1,2])
plt.show()


# <h3>Satisfaction </h3>
# <br>
# Employees with a low level of satisfaction (<0.3) = 1941
# Employees with a high level of satisfaction (=>0.7) = 6502
# <br>
# <h3>Last Evaluation</h3>
# <br>
# Considering a grade equal to or greater than 7 as high, we have 8015 employees well evaluated by the company i.e. around 53%
# <br>
# <h3>Number of Project</h3>
# <br>
# Most employees worked on 3-4 projects and the employees with 7 Projects are very less in number. The employees worked on at least 2 and at most 7 projects.
# <br>
# <h3>Time Spend</h3><br> 
# This is a positively skewed distribution, with a reduction in the number of employees per year as it approaches the right tail.
# <br>
# <h3>Promotions</h3><br> Very few people promoted. Only 2% of people have already been promoted. Knowing the career plan policy would be important to understand this shortage of promotions.

# In[21]:


fig = plt.figure(figsize=(15,7))
sns.barplot(x='left', y ='satisfaction_level' ,data=df)
plt.show()


# When we compare the values between the hired and those who leave, we can see that, for the hired the satisfaction ranges that prevail are those of satisfaction and regular satisfaction, the two corresponding to 87% of the hired. <br> <br>For those who leave the dissatisfied range alone corresponds to 49% of this set. <b> Hence, almost half of the employees who leave the company are dissatisfied.</b>
# 
# Among the hired we observe that the satisfied ones form the largest group, followed by those of regular satisfaction, dissatisfied individuals are the minority.<br>
# 
# Rename axes

# ### Did the employees who left receive a low salary?

# In[22]:


fig,ax = plt.subplots(2,1, figsize=(17,12))
sns.countplot(x='salary',data=df , ax=ax[0])
sns.countplot(x='salary', hue = 'left' , data=df , ax=ax[1])
plt.show()


# ### Employees in each Department

# In[23]:


fig,ax = plt.subplots(2,1, figsize=(17,12))
sns.countplot(x='Department',data=df , ax=ax[0])
sns.countplot(df['Department'],hue=df['left'],data=df , ax=ax[1])
plt.show()


# The Sales department is the one with the largest number of employees, followed by the Technical and Support department, totaling 9,089 employees. Management is the smallest of them with 630 employees.

# ### Department and left employees in relation with the salary range

# In[24]:


fig = plt.figure(figsize=(15,7))
sns.barplot(y='left',x='Department', hue= 'salary',data=df)
plt.show()


# Add observation

# ### Employees of which department left the company in maximum numbers?

# In[25]:


fig,ax = plt.subplots(2,1, figsize=(17,12))
sns.countplot(df['Department'],hue=df['left'],data=df , ax=ax[0])
sns.boxplot(y='average_montly_hours', x='Department', hue= 'left',data=df , ax=ax[1])
plt.show()


# The employees who left work for longer hours and lesser salaries.

# In[26]:


fig = plt.figure(figsize=(15,7))
sns.boxplot(y='average_montly_hours',x='time_spend_company', hue= 'left',data=df)
plt.show()


# Here we can see the people who have worked in the company between 2 and 10 years. <br><br> For those who worked for a total of 2 years, the difference is small between the hired and who leaves. The employee who leaves worked for more hours. <br><br>The data for those who worked for 3 years is quite different from the others, they exhibit a new grouping of people who leaves and worked less hours than the average monthly.<br><br> During the univariate graphs section, seeing the distribution of the total number of people per years worked, observe a peak in the 3rd year and from there a fall in the total of people per year after the 4th year.<br> look at outliers

# ### Is the employee's dissatisfaction a factor for leaving the company? 

# In[27]:


fig,ax = plt.subplots(2,1, figsize=(17,12))
sns.boxplot(y='satisfaction_level',x='time_spend_company', hue= 'left',data=df , ax=ax[0])
sns.boxplot(y='last_evaluation',x='time_spend_company', hue= 'left',data=df , ax=ax[1])
plt.show()


# We see dissatisfaction with employees who leave the company in the 3rd or 4th year. People who left the company in the 5th grade, mostly showed low or high satisfaction. Most of then who worked at the company for 6 years demonstrated a high satisfaction level of over 0.75. 
# <br><br>
# Again the 3rd year is a major factor. The people who left the company in the 3rd year received a low score in the last evaluation, a fact that does not repeat from the 4th to the 6th year, where the majority of those employees received grades higher than 0.8.

# ### Does the salary received depend upon the number of projects done?

# In[28]:


fig = plt.figure(figsize=(15,7))
sns.countplot(x='salary', hue = 'number_project',data=df)
plt.show()


# Both people with high, medium or low salary worked on 2 projects or more. Around 140 people worked on 7 projects and received a low salary. So answering the question, there is no evidence of connection between number of projects and salary range.

# ### Did Employees who were involved in more projects received more?

# In[29]:


fig = plt.figure(figsize=(15,7))
sns.boxplot(y='last_evaluation',x='number_project', hue= 'salary',data=df)
plt.show()


# People who left and also worked on 7 projects, and received low scores would have been dissatisfied. Those who left the company involved in 2 projects, regardless of the salary range, predominantly received low scores in their evaluations. People with more than 4 projects tended to receive better grades in their assessments.<br><br>outliers

# In[30]:


fig = plt.figure(figsize=(15,7))
sns.boxplot(y='satisfaction_level',x='number_project', hue= 'salary',data=df)
plt.show()


# In this plot we have the proportion of the satisfaction of who left by numbers of projects. <br><br>People with 2 projects showed a low to regular satisfaction. The group with 6 or 7 projects also showed a low to regular satisfaction level. <br><br> Employees who worked on 4 or 5 projects had a predominantly high satisfaction.
# 

# ### Is Work Load a reason for employees to leave?

# In[31]:


# Find the effect of satisfaction level and the average monthly hours with department and salary level
# on departure of  employees.
plt.figure(figsize=(9,9))
sns.relplot(x="satisfaction_level",
                y="average_montly_hours",
                col="Department",
                hue="salary",
                kind="scatter",
                height=10,
                aspect=0.3,
                data=df[df['left']==1])


# In[32]:


plt.figure(figsize=(9,9))
sns.relplot(x="last_evaluation",
                y="average_montly_hours",
                col="Department",
                hue="salary",
                kind="scatter",
                height=10,
                aspect=0.3,
                data=df[df['left']==1])


# ### Work Accident

# In[33]:


df['Work_accident'].value_counts()


# In[34]:


fig = plt.figure(figsize=(15,7))
sns.countplot(x='Work_accident', hue = 'left',data=df)
plt.show()


# Both people with high, medium or low salary worked on 2 projects or more. 144 people worked on 7 projects and received a low salary. So there is no proof that the larger the number of projects, the better the salary range.
# 
# Even though they have suffered an accident at work, the employees remain in the company. <b> This variable does not seem to be related to the employee's exit.</b>
# 
# Now that we know the importance of the variables satisfaction_level and average_montly_hours we will calculate the value of the pearson correlation for the data set variables hr:

# ### Correlation Matrix

# In[35]:


fig = plt.figure(figsize=(15,7))
cor_mat=num_data.corr()
sns.heatmap(cor_mat ,annot = True, cmap='Blues')
plt.show()

# cmap green yellow red


# The darker, the more correlated are the variables, the whites have negative correlation values and the positive blue ones.
# 
# For the left variable the correlations were found: <b>
# 
# - Satisfaction_level = With a value of -0.39 that is variable with the strongest correlation with left
# - Work_accident = Poor correlation, value -0.15
# - Time_spend_company = Poor correlation, value 0.14
# - Average_montly_hours = Very poor correlation, value 0.071
# - last_evaluation = Very weak correlation, value 0.0066
# - number_project = Very poor correlation, value 0.024
# - Promotion_last_5years = Very weak correlation, value -0.062

# In[36]:


fig = plt.figure(figsize=(15,7))
sns.scatterplot(x="average_montly_hours", y= "satisfaction_level",hue='left' ,data=df)
plt.show()


# <h3>Summary of the Observations</h3> <br>
# The people who left the company were <b>overburdened and unsatisfied.</b>
# 
# - The unsatisfied who worked less than the general average and those who worked more than the average and more than the remain;
# 
# - Those with a good level of satisfaction, but who also had a monthly average of hours worked over 201 hours of the general average.

# ## Data Preprocessing
# Convert the salary column to categorical

# In[37]:


df.salary=df.salary.astype('category')
df.salary=df.salary.cat.reorder_categories(['low', 'medium', 'high'])
df.salary = df.salary.cat.codes


# In[38]:


departments = pd.get_dummies(df.Department)
departments.head(5)


# In[39]:


departments = departments.drop("accounting", axis=1)
df = df.drop("Department", axis=1)
df = df.join(departments)
df.head(5)


# As the values in the column satisfaction_level and last_evaluation are not in the order of the other entries we can multiply the values in the column by a constant to make it in the order of the column values.

# In[40]:


# Multiplying by 10, it won't change the value but it is useful to visualize
df.satisfaction_level=df.satisfaction_level*10
df.last_evaluation=df.last_evaluation*10


# ### Percentage of Employee Attrition

# In[41]:


n_employees = len(df)
print(df.left.value_counts())
print(df.left.value_counts()/n_employees*100)


# 11,428 employees stayed, which accounts for about 76% of the total employee count. Similarly, 3,571 employees left, which accounts for about 24% of them

# In[42]:


def count_target_plot(data,target):
    plt.figure(figsize=(8,8))
    ax=sns.countplot(data=data,x=data[target],order=data[target].value_counts().index)
    plt.xlabel('Target Variable- Left')
    plt.ylabel('Distribution of target variable')
    plt.title('Distribution of Left')
    total = len(data)
    for p in ax.patches:
            ax.annotate('{:.1f}%'.format(100*p.get_height()/total), (p.get_x()+0.1, p.get_height()+5))


# In[43]:


count_target_plot(df,'left')


# ###  Separating Target and Features
# Let us separate the Dependent Variable (target) and the Independent Variables (predictors).

# In[44]:


target=df.left
features=df.drop('left',axis=1)


# ### Splitting the Dataset
# We will split both target and features into train and test sets with 80% and 20% ratio, respectively.

# In[45]:


splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
for train_index, test_index in splitter.split(features, target):
        features_train, features_test = features.iloc[train_index], features.iloc[test_index]
        target_train, target_test = target.iloc[train_index], target.iloc[test_index]


# ## Model Building

# ### Logistic Regression

# In[46]:


logr = LogisticRegression(random_state=42)
logr.fit(features_train, target_train)
y_pred_logr = logr.predict(features_test)
print("Score on Train data : " , logr.score(features_train,target_train)*100)
print("Score on Test data : " , accuracy_score(target_test, y_pred_logr)*100)


# ### Decision tree 

# In[47]:


DT = DecisionTreeClassifier(random_state=42)
DT.fit(features_train, target_train)
y_pred_DT = DT.predict(features_test)
print("Score on Train data : " , DT.score(features_train,target_train)*100)
print("Score on Test data : " , accuracy_score(target_test, y_pred_DT)*100)


# The accuracy is 100% on training data and the model is overfitting. So we will purne the tree, by setting the maximum depth and limiting the sample size.

# In[48]:


DT1 = DecisionTreeClassifier(max_depth=9,min_samples_leaf=2, random_state=42)
DT1.fit(features_train, target_train)
y_pred_DT1 = DT1.predict(features_test)
print("Score on Train data : " , DT1.score(features_train,target_train)*100)
print("Score on Test data : " , accuracy_score(target_test, y_pred_DT1)*100)


# ### Random Forest

# In[49]:


RF = RandomForestClassifier(random_state=42)
RF.fit(features_train, target_train)
y_pred_RF = RF.predict(features_test)
print("Score on Train data : " , RF.score(features_train,target_train)*100)
print("Score on Test data : " , accuracy_score(target_test, y_pred_RF)*100)


# Again the model is overfitting, hence we will set the maximum depth, limiting the sample size, and maximum features.
# 

# In[50]:


RF1 = RandomForestClassifier(min_samples_leaf=1, max_features= 5, max_depth=9,random_state=42)
RF1.fit(features_train, target_train)
y_pred_RF1 = RF.predict(features_test)
print("Score on Train data : " , RF1.score(features_train,target_train)*100)
print("Score on Test data : " , accuracy_score(target_test, y_pred_RF1)*100)

# Lasso, regualrization, Hyp Tuning for overfitting


# ### KNN

# In[51]:


knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(features_train, target_train)
y_pred_knn = knn.predict(features_test)
print("Score on Train data : " , knn.score(features_train,target_train)*100)
print("Score on Test data : " , accuracy_score(target_test, y_pred_knn)*100)


# In[52]:


print("Logistic Regression : ")
print(classification_report(target_test,y_pred_logr))
print("=======================================================")
print("Decison Tree : ")
print(classification_report(target_test,y_pred_DT))
print(classification_report(target_test,y_pred_DT1))
print("=======================================================")
print("Random Forest : ")
print(classification_report(target_test,y_pred_RF))
print(classification_report(target_test,y_pred_RF1))
print("=======================================================")
print("KNN : ")
print(classification_report(target_test,y_pred_knn))


# In[53]:


pred_prob1 = logr.predict_proba(features_test)
pred_prob2 = DT.predict_proba(features_test)
pred_prob3 = RF.predict_proba(features_test)
pred_prob4 = knn.predict_proba(features_test)

from sklearn.metrics import roc_curve

# roc curve for models
fpr1, tpr1, thresh1 = roc_curve(target_test, pred_prob1[:,1], pos_label=1)
fpr2, tpr2, thresh2 = roc_curve(target_test, pred_prob2[:,1], pos_label=1)
fpr3, tpr3, thresh3 = roc_curve(target_test, pred_prob3[:,1], pos_label=1)
fpr4, tpr4, thresh4 = roc_curve(target_test, pred_prob4[:,1], pos_label=1)

random_probs = [0 for i in range(len(target_test))]
p_fpr, p_tpr, _ = roc_curve(target_test, random_probs, pos_label=1)


# In[54]:


from sklearn.metrics import roc_auc_score

# auc scores
auc_score1 = roc_auc_score(target_test, pred_prob1[:,1])
auc_score2 = roc_auc_score(target_test, pred_prob2[:,1])
auc_score3 = roc_auc_score(target_test, pred_prob3[:,1])
auc_score4 = roc_auc_score(target_test, pred_prob4[:,1])

print(auc_score1, auc_score2, auc_score3, auc_score4)


# In[55]:


import matplotlib.pyplot as plt
plt.style.use('seaborn')

# plot roc curves
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='Logistic Regression')
plt.plot(fpr2, tpr2, linestyle='--',color='green', label='Decision Tree')
plt.plot(fpr3, tpr3, linestyle='--',color='red', label='Random Forest')
plt.plot(fpr4, tpr4, linestyle='--',color='yellow', label='KNN')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

plt.legend(loc='best')
plt.savefig('ROC',dpi=500)
plt.show();


# <b>AUC Score for Logistic Regression:</b> 0.8248766698280362 <br>
# <b>AUC Score for Random Forests:</b> 0.9743138725306396 <br>
# <b>AUC Score for Decision Trees:</b> 0.9919011349071563<br>
# <b>AUC Score for KNN:</b> 0.9769688715381165<br>

# <h1>=============================================================</h1>
# 

# <h2>                     End of Project </h2><br>
# <h1>=============================================================</h1>

# In[230]:


import statsmodels.api as sm

# Logistic regression model
logm1 = sm.GLM(target_train,(sm.add_constant(features_train)), family = sm.families.Binomial())
logm1.fit().summary()


# In[231]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

from sklearn.feature_selection import RFE
rfe = RFE(logreg, 15)             # running RFE with 15 variables as output
rfe = rfe.fit(features_train, target_train)

list(zip(features_train.columns, rfe.support_, rfe.ranking_))


# In[232]:


col = features_train.columns[rfe.support_]

features_train_sm = sm.add_constant(features_train[col])
logm2 = sm.GLM(target_train,features_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[233]:


# Getting the predicted values on the train set
target_train_pred = res.predict(features_train_sm)
target_train_pred[:10]


# In[234]:


target_train_pred_final = pd.DataFrame({'Churn':target_train.values, 'Churn_Prob':target_train_pred})
target_train_pred_final['CustID'] = target_train.index
target_train_pred_final.head()


# In[235]:


target_train_pred_final['predicted'] = target_train_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the dataframe
target_train_pred_final.head()


# In[236]:


from sklearn import metrics

# Confusion matrix 
confusion = metrics.confusion_matrix(target_train_pred_final.Churn, target_train_pred_final.predicted )
print(confusion)


# In[237]:


print(metrics.accuracy_score(target_train_pred_final.Churn, target_train_pred_final.predicted))


# In[238]:


# vif = pd.DataFrame()
# vif['Features'] = features_train[col].columns
# vif['VIF'] = [variance_inflation_factor(features_train[col].values, i) for i in range(features_train[col].shape[1])]
# vif['VIF'] = round(vif['VIF'], 2)
# vif = vif.sort_values(by = "VIF", ascending = False)
# vif


# In[239]:


# col = col.drop('last_evaluation', 1)

# Let's re-run the model using the selected variables
# features_train_sm = sm.add_constant(features_train[col])
# logm3 = sm.GLM(target_train,features_train_sm, family = sm.families.Binomial())
# res = logm3.fit()
# res.summary()


# In[ ]:





# In[ ]:




