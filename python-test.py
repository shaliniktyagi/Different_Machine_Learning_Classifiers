#!/usr/bin/env python
# coding: utf-8

# # Python Test
# You have been provided with learning activity data collected from a global sample of students. The attributes are broken down into demographic features like gender and nationality, academic backgrounds like educational stage and grade level, and behavioral observations like engagement counts and parent satisfaction. 
# 
# The students are classified into three ability levels based on how they perform on selected material: 
# 
# __Low__: grades ranging from 0-39 <br>
# __Medium__: grades ranging from 40-89 <br>
# __High__: grades ranging from 90-100 <br>
# 
# You'll have to dive into the data, learn the attributes, and construct a model that can predict the performance level of a given student.  Some common libraries have been loaded, but use any more that you wish.

# # Load the Data
# Load the data into a pandas dataframe. Display the shape of the data, and count the number of null values in each column.

# In[148]:


# load the packages
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


# In[149]:


# Load the dataset
data = pd.read_csv('data.csv')
data.head()


# In[150]:


# shape of the dataset
print(data.shape)

# count the number of missing values (null) in each column
print(data.isna().sum())


# # Perform basic EDA
# Use a library of your choice to visualise attributes in the data. Answer each question below with a visualisation. The attribute glossary has also been provided for your reference.
# 
# ## Attribute Glossary
# Gender- student's gender (nominal: 'Male' or 'Female’)
# 
# Nationality- student's nationality (nominal:' Luxembourg',' Lebanon',' Egypt',' USA',' Nigeria',' Venezuela',' Rwanda',' Germany',' France',' India',' Cambodia',' Chile',' Spain')
# 
# Place of birth- student's Place of birth (nominal:' Luxembourg',' Lebanon',' Egypt',' USA',' Nigeria',' Venezuela',' Rwanda',' Germany',' France',' India',' Cambodia',' Chile',' Spain')
# 
# Educational Stages- educational level student belongs (nominal:‘lowerlevel’,’MiddleSchool’,’HighSchool’)
# 
# Grade Levels- grade to which the student belongs (nominal:‘G-01’, ‘G-02’, ‘G-03’, ‘G-04’, ‘G-05’, ‘G-06’, ‘G-07’, ‘G-08’, ‘G-09’, ‘G-10’, ‘G-11’, ‘G-12‘)
# 
# Section ID- classroom student belongs (nominal:’A’,’B’,’C’)
# 
# Topic- course topic (nominal:'Mathematics",'English','Science','Geography','History','French','Spanish','Sport Science’)
# 
# Semester- school year semester (nominal:’First’,’Second’)
# 
# Parent responsible for student (nominal:’mother’,’father’)
# 
# Raised hand- how many times the student raises his/her hand in the classroom (numeric, discrete)
# 
# Used resources- how many times the student uses course content (numeric, discrete)
# 
# Viewed notifications- how many times the student checks their new notifications (numeric, discrete)
# 
# Discussion groups- how many times the student participate on discussion groups (numeric, discrete)
# 
# Parent Answered Survey- parent answered the surveys which are provided from school or not (nominal:’Yes’,’No’)
# 
# Parent School Satisfaction- the degree of parental satisfaction towards the school (nominal:’Yes’,’No’)
# 
# Student Absence Days- the number of absence days for each student (nominal:above-7, under-7)

# ## Show the relation of each column to students' ability level
# 
# Somehow indicate the columns with a positive correlation or relationship to 'abilityLevel'. This could be via a table or a visualisation.

# In[151]:


# fill the missing values with 0
data1 = data.fillna(0) 

# Delete parentSchoolSatisfaction column from the dataframe
data1[['gradeG','gradeLevel']] = data1.gradeLevel.str.split("-",expand=True,)
data2 = data1.drop(['parentSchoolSatisfaction','studentAbsenceDays','gradeG', 'parent'], axis = 1) 

data2.info()
data2


# In[152]:


#Diasplay the classes distribution in the data
sns.countplot(x = "abilityLevel", data = data2, palette = "Greens");
plt.show()


# In[153]:


# visualize ablilty level and raised Hand 
plt.figure(figsize=(10,6))
plt.title('relation between ablilty level and raise hand')
sns.barplot(x="abilityLevel", y="raisedHand", data=data2)
plt.show()


# In[154]:


plt.figure(figsize=(12,6))
plt.title('The relationship between topic and nationality')
sns.countplot(x='topic', hue = 'nationality', data=data2)
plt.show()


# In[155]:


plt.figure(figsize=(12,6))
sns.countplot(x='topic', hue = 'abilityLevel', data=data2)
plt.show()


# In[169]:


fig = plt.figure()
# Create an axes instance
#ax = fig.add_axes([0,0,1,1])
# Create the boxplot
bp = plt.boxplot(data2['topic'])
plt.show()


# In[156]:




sns.set_style("ticks")
sns.pairplot(data2,hue = 'abilityLevel',diag_kind = "kde",palette = "husl")
plt.show()


# # Model the data

# Answer the questions below with whatever machine learning libraries you choose. The goal here is to predict the 'abilityLevel' attribute for every student.

# ## Prepare the data
# Transform the data in any way you choose. Be ready to explain your reasoning for selecting the columns that need to be transformed as well as the transformations applied. When all transformations are applied, split the data as necessary for prediction.

# In[157]:



# con
ord_enc = OrdinalEncoder()
data2["abilityLevel_code"] = ord_enc.fit_transform(data2[["abilityLevel"]])
data2["gender_code"] = ord_enc.fit_transform(data2[["gender"]])
data2["nationality_code"] = ord_enc.fit_transform(data2[["nationality"]])
data2["placeOfBirth_code"] = ord_enc.fit_transform(data2[["placeOfBirth"]])
data2["educationalStage_code"] = ord_enc.fit_transform(data2[["educationalStage"]])
data2["topic_code"] = ord_enc.fit_transform(data2[["topic"]])
data2["sectionId_code"] = ord_enc.fit_transform(data2[["sectionId"]])
data2["semester_code"] = ord_enc.fit_transform(data2[["semester"]])
data2["parentAnsweredSurvey_code"] = ord_enc.fit_transform(data2[["parentAnsweredSurvey"]])

#data2["gradeLevel_code"] = ord_enc.fit_transform(data2[["gradeLevel"]])
#data2["parent_code"] = ord_enc.fit_transform(data2[["parent"]])
data2.head(15)


# In[158]:


data3 = data2.drop(['abilityLevel','gender','nationality', 'placeOfBirth', 'educationalStage', 'topic', 'sectionId', 'semester', 'parentAnsweredSurvey','placeOfBirth_code'], axis = 1)
data3.head(15)


# In[159]:


data3.describe()


# In[160]:


plt.figure(figsize=(20,10))
plt.title('correlation matrix')
sns.heatmap(data3.corr(), cmap = 'RdYlGn', annot = True )
plt.show()


# In[161]:


# class distribution
print(data3.groupby('abilityLevel_code').size())
data4 = data3.dropna()
data4.isna().sum()


# In[162]:


# split the dataset into training and test data
labels = data4['abilityLevel_code']
# Remove the labels from the features
features= data4.drop('abilityLevel_code', axis = 1)
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.30, random_state = 1)
print( train_features.shape, train_labels.shape)
print (test_features.shape, test_labels.shape)
test_features.head()
print(test_features.head())
print(test_labels.head())


# In[163]:


# transform the dataset
oversample = SMOTE()
features, labels = oversample.fit_resample(features, labels)
# summarize distribution
counter = Counter(labels)
for k,v in counter.items():
	per = v / len(labels) * 100
	print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
# plot the distribution
plt.bar(counter.keys(), counter.values())
plt.show()


# In[164]:


train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.30, random_state = 1)
print( train_features.shape, train_labels.shape)
print (test_features.shape, test_labels.shape)
#test_features.head()
#print(test_features.head())
#print(test_labels.head())


# ### Building Models:
# 

# In[96]:


# Check Algorithms which one work on the given data
models = []

models.append(('RFC', RandomForestClassifier()))
models.append(('NB', GaussianNB()))
models.append(('KNN', KNeighborsClassifier()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, train_features,train_labels, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# ## Predict student level
# Feed your input data into a model of your choice and observe how it performs. Be ready to explain why you selected this model for your experiment and the metric you used to evaluate performance.

# In[99]:


# Make predictions on validation dataset
model = RandomForestClassifier()
model.fit(train_features,train_labels)
predictions = model.predict(test_features)


# In[100]:


# Evaluate predictions
print(accuracy_score(test_labels, predictions))
print(confusion_matrix(test_labels, predictions))
print(classification_report(test_labels, predictions))


# ## Tune your hyper-parameters
# Demonstrate a tuning of hyper-parameters that will improve performance. You can do this via manual testing or a programatic package. Be ready to explain why those parameters affected model performance.

# In[101]:


model1= RandomForestClassifier()
n_estimators = [10, 100, 1000]
max_features = ['sqrt', 'log2']
# define grid search
grid = dict(n_estimators=n_estimators, max_features=max_features)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model1, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(train_features,train_labels)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[102]:


best_model =grid_result.best_estimator_
print(best_model)

predictions = best_model.predict(test_features)


# In[165]:


print(accuracy_score(test_labels, predictions))
print(confusion_matrix(test_labels, predictions))
print(classification_report(test_labels, predictions))


# # Extra thoughts for discussion
# 
# What other models might you have chosen?<br>
# What more information might be relevant to classify students?<br>
# If an accurate model is developed, how might it be used to improve a student's experience using CENTURY?

# In[237]:




