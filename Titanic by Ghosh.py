#!/usr/bin/env python
# coding: utf-8

# In[62]:


import pandas as pd
import numpy as np
import seaborn  as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# # Loading data

# Train and Test Data is loaded and being cross-checked.

# In[122]:


train_data = pd.read_csv('train.csv')
train_data.head()


# In[123]:


test_data = pd.read_csv('test.csv')
test_data.head()


# ## Summerizing the Data Set

# In[124]:


# how many rows and columns?

train_data.shape


# In[125]:


#how many survived and how many died?

train_data.groupby ('Survived').size()


# # Data Modelling

# In[126]:


train_data.isnull().head()


# In[127]:


sns.heatmap(train_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# ###### If we glimpse at the data, we are having multiple missing information mainly,  age information,  a lot of cabin info and we're missing one row of embarked. We'll come back to this problem of missing data a little later datacleansing part. But before that lets focus on some exploratory data analysis on a visual level.

# In[128]:


sns.set_style('whitegrid')
sns.countplot(y='Survived',data=train_data)
sns.countplot(y='Survived',data=train_data,hue='Sex',palette='RdBu_r')


# ###### As we can see we clearly have a trend here. It looks like people that did not survive were much more likely to be men. While those who survived were twice as likely to be female

# In[129]:


sns.countplot(y='Survived',data=train_data,hue='Pclass')


# ###### Also it looks like the people who did not survive were overwhelmingly part of 3rd class. People that did survive were from the higher classes. Now lets try and understand the age of the onboard passengers.

# In[130]:


sns.distplot(train_data['Age'].dropna(),bins=30,kde=False)


# ###### There seems to be an interesting bi-modal distribution where there are quite a few young passengers between age 0 and 10. Then the average age tends to be around 20-30.

# In[131]:


sns.countplot(y='SibSp',data=train_data)


# In[132]:


train_data['Fare'].hist(bins=40,figsize=(10,4))


# # Data Cleaning

# ###### As per the given data we have missing values in Age, cabin and embarked 
# ###### column. We wont be considering cabin column anymore. Here we will be carrying 
# ###### out Data cleansing by droping cabin and replacing the missing values in 
# ###### other two column with mean vales.
# 
# ###### Hence we will drop Ticket, Fare and Cabin as they are not helpful in this regard.

# In[133]:


to_drop = ['PassengerId',
           'Ticket',
           'Fare',
           'Cabin',
           'Embarked',
           'Name']

train_data.drop(to_drop, inplace=True, axis=1)


# In[134]:


to_drop = ['Ticket',
           'Fare',
           'Cabin',
           'Embarked',
           'Name']

test_data.drop(to_drop, inplace=True, axis=1)


# In[135]:


plt.figure(figsize=(10,7))
sns.boxplot(x='Pclass',y='Age',data=train_data)


# ###### Filling all the empty values with mean of the column. 
# 
# ###### same to do with both Train and Test cases.

# In[136]:


train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())

print(train_data)


# In[137]:


train_data.replace({'Sex' : {'male':0, 'female':1}}, inplace = True)

train_data.info()
train_data.head()


# In[138]:


train_data.loc[ train_data['Age'] <= 16, 'Age'] = 0
train_data.loc[(train_data['Age'] > 16) & (train_data['Age'] <= 32), 'Age'] = 1
train_data.loc[(train_data['Age'] > 32) & (train_data['Age'] <= 48), 'Age'] = 2
train_data.loc[(train_data['Age'] > 48) & (train_data['Age'] <= 64), 'Age'] = 3
train_data.loc[ train_data['Age'] > 64, 'Age']                           = 4


# In[139]:



x = train_data.drop("Survived", axis = 1)
y = train_data['Survived']

x.info()


# In[140]:


test_data.loc[ train_data['Age'] <= 16, 'Age'] = 0
test_data.loc[(train_data['Age'] > 16) & (test_data['Age'] <= 32), 'Age'] = 1
test_data.loc[(train_data['Age'] > 32) & (test_data['Age'] <= 48), 'Age'] = 2
test_data.loc[(train_data['Age'] > 48) & (test_data['Age'] <= 64), 'Age'] = 3
test_data.loc[ train_data['Age'] > 64, 'Age']    


# In[141]:


test_data.replace({'Sex' : {'male':0, 'female':1}}, inplace = True)

test_data.info()
test_data.head()


# ### Logistic Regression

# In[142]:


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

t_s = np.linspace(0.1,0.4,4)
r_s = np.linspace(0,100,11).astype(int)


log_cols = ["Test Size", "Ramdon Size", "Accuracy"]
log 	 = pd.DataFrame(columns=log_cols)

n_splits = 10
acc_dict = {}

for i in t_s:
    for j in r_s:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = i, random_state = j)
        x_train.shape, x_test.shape
        logreg = LogisticRegression()
        logreg.fit(x_train, y_train)
        predictions = logreg.predict(x_test)
        acc = accuracy_score(y_test, predictions)
        #print(classification_report(y_test, predictions))
        #print(logreg.score(x_train,y_train))
        #print(logreg.score(x_test,y_test))
        #train_pred = train_pred.append(logreg.score(x_train,y_train))
        #test_pred = test_pred.append(logreg.score(x_test,y_test))
        
        
        
        if i in acc_dict:
            if j in acc_dict[i]:
                acc_dict[i][j] += acc
            else:
                acc_dict[i][j] = acc
        else:
            acc_dict[i] = {}
            acc_dict[i][j] = acc
            
        #print(acc_dict)

for i in acc_dict:
    for j in acc_dict[i]:
        acc_value = acc_dict[i][j] / n_splits * 1000
        log_entry = pd.DataFrame([[i, j, acc_value]], columns=log_cols)
        log = log.append(log_entry)
        
#print(log)

#plt.scatter(i, acc_value)

#plt.show()

plt.figure()

plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')

heatmap_data = log.pivot("Test Size", "Ramdon Size", "Accuracy")
sns.set(rc = {'figure.figsize':(15,8)})
ax = sns.heatmap(heatmap_data, annot=True, linewidths=5, fmt='0.3f')


# In[143]:


from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test, predictions)
sns.heatmap(matrix, annot = True, cmap = 'Blues', fmt = 'd')


# In[144]:


logreg.score(x_train,y_train)


# In[145]:


logreg.score(x_test,y_test)


# ### Cross-validation

# In[146]:


from sklearn.model_selection import cross_validate

scores = cross_validate(logreg, x, y, scoring='accuracy', cv=10)
print(scores['test_score'])


# In[147]:


scores = pd.Series(scores)
print(scores.test_score.mean())


# ### Support Vector Model

# In[148]:


from sklearn.svm import SVC

svm = SVC()
svm.fit(x_train, y_train)
print(svm.score(x_train, y_train))
print(svm.score(x_test, y_test))


# ## Making Predictions

# ##### Here we are taking out Passenger Id from the table as we need it to predict who survived and who died. In the final prediction we are having two columns having Passenger ID, Survived.

# In[158]:


test_x = test_data.drop('PassengerId',axis=1)


# In[159]:


predictions = logreg.predict(test_x)


# In[160]:


final_prediction = pd.DataFrame({'PassengerId':test_data['PassengerId'],'Survived':predictions})


# In[161]:


final_prediction.head()


# In[162]:


sns.countplot(y='Survived',data=final_prediction)


# In[ ]:




