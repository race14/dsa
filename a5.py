#!/usr/bin/env python
# coding: utf-8

# Implement K-Nearest Neighbors algorithm on diabetes.csv dataset. Compute confusion
# matrix, accuracy, error rate, precision and recall on the given dataset.

# In[2]:


import pandas as pd
import seaborn as sns


# In[3]:


df=pd.read_csv('diabetes.csv')


# In[4]:


df


# In[5]:


df.shape


# In[6]:


df.columns


# In[7]:


df.head()


# In[8]:


df.describe()


# In[9]:


#input data
x= df.drop('Outcome', axis=1)

#Output Data 
y=df['Outcome']


# In[10]:


sns.countplot(x=y)


# In[11]:


y.value_counts()


# In[12]:


#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_scaled=scaler.fit_transform(x)


# In[13]:


#Cross-Validation
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(
    x_scaled,y,random_state=0,test_size=0.25
)


# In[14]:


x.shape


# In[15]:


x_train.shape


# In[16]:


x_test.shape


# In[17]:


from sklearn.neighbors import KNeighborsClassifier


# In[18]:


knn= KNeighborsClassifier(n_neighbors=33)


# In[19]:


knn.fit(x_train,y_train)


# In[20]:


from sklearn.metrics import accuracy_score,ConfusionMatrixDisplay
from sklearn.metrics import classification_report


# In[21]:


y_pred=knn.predict(x_test)


# In[22]:


ConfusionMatrixDisplay.from_predictions(y_test,y_pred)


# In[23]:


print(classification_report(y_test,y_pred))


# In[24]:


accuracy_score(y_test,y_pred)

