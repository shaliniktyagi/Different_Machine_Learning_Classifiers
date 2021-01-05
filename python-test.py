


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


# Load the dataset
data = pd.read_csv('data.csv')
data.head()

# shape of the dataset
print(data.shape)

# count the number of missing values (null) in each column
print(data.isna().sum())
# fill the missing values with 0
data1 = data.fillna(0) 

# Delete parentSchoolSatisfaction column from the dataframe
data1[['gradeG','gradeLevel']] = data1.gradeLevel.str.split("-",expand=True,)
data2 = data1.drop(['parentSchoolSatisfaction','studentAbsenceDays','gradeG', 'parent'], axis = 1) 

data2.info()
data2

#Diasplay the classes distribution in the data
sns.countplot(x = "abilityLevel", data = data2, palette = "Greens");
plt.show()

# visualize ablilty level and raised Hand 
plt.figure(figsize=(10,6))
plt.title('relation between ablilty level and raise hand')
sns.barplot(x="abilityLevel", y="raisedHand", data=data2)
plt.show()

plt.figure(figsize=(12,6))
plt.title('The relationship between topic and nationality')
sns.countplot(x='topic', hue = 'nationality', data=data2)
plt.show()


plt.figure(figsize=(12,6))
sns.countplot(x='topic', hue = 'abilityLevel', data=data2)
plt.show()


fig = plt.figure()
# Create an axes instance
#ax = fig.add_axes([0,0,1,1])
# Create the boxplot
bp = plt.boxplot(data2['topic'])
plt.show()




sns.set_style("ticks")
sns.pairplot(data2,hue = 'abilityLevel',diag_kind = "kde",palette = "husl")
plt.show()


# # Model the data
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



data3 = data2.drop(['abilityLevel','gender','nationality', 'placeOfBirth', 'educationalStage', 'topic', 'sectionId', 'semester', 'parentAnsweredSurvey','placeOfBirth_code'], axis = 1)
data3.head(15)


data3.describe()




plt.figure(figsize=(20,10))
plt.title('correlation matrix')
sns.heatmap(data3.corr(), cmap = 'RdYlGn', annot = True )
plt.show()


# class distribution
print(data3.groupby('abilityLevel_code').size())
data4 = data3.dropna()
data4.isna().sum()



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



# Make predictions on validation dataset
model = RandomForestClassifier()
model.fit(train_features,train_labels)
predictions = model.predict(test_features)


# Evaluate predictions
print(accuracy_score(test_labels, predictions))
print(confusion_matrix(test_labels, predictions))
print(classification_report(test_labels, predictions))


# ## Tune your hyper-parameter

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


best_model =grid_result.best_estimator_
print(best_model)

predictions = best_model.predict(test_features)


print(accuracy_score(test_labels, predictions))
print(confusion_matrix(test_labels, predictions))
print(classification_report(test_labels, predictions))


