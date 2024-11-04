#!/usr/bin/env python
# coding: utf-8

# Given a bank customer, build a neural network-based classifier that can determine whether they will leave or not in the next 6 months.
# Dataset Description: The case study is from an open-source dataset from Kaggle. The dataset contains 10,000 sample points with 14 distinct features such as CustomerId, CreditScore, Geography, Gender, Age, Tenure, Balance, etc. Link to the Kaggle project: https://www.kaggle.com/barelydedicated/bank-customer-churn-modeling Perform following steps: 1. Read the dataset. 2. Distinguish the feature and target set and divide the data set into training and test sets. 3. Normalize the train and test data. 4. Initialize and build the model. Identify the points of improvement and implement the same. 5. Print the accuracy score and confusion matrix (5 points).

# In[1]:


import pandas as pd
import seaborn as sns 


# In[2]:


df = pd.read_csv('Churn_Modelling.csv')


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


#input data
x= df[['CreditScore','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']]

#Output Data
y=df['Exited']



# In[6]:


sns.countplot(x=y)


# In[7]:


y.value_counts()


# In[8]:


get_ipython().system('pip install imbalanced-learn')


# In[9]:


from imblearn.over_sampling import RandomOverSampler


# In[10]:


ros = RandomOverSampler(random_state=0)


# In[11]:


x_res, y_res = ros.fit_resample(x,y)


# In[12]:


y_res.value_counts()


# In[13]:


# Normalize
from sklearn.preprocessing import StandardScaler


# In[14]:


scaler = StandardScaler()


# In[15]:


x_scaled = scaler.fit_transform(x_res)


# In[16]:


x_scaled


# In[17]:


#Cross-validation
from sklearn.model_selection import train_test_split


# In[18]:


x_train,x_test,y_train,y_test = train_test_split(x_scaled,y_res, random_state=0,test_size=0.25)


# In[19]:


x_res.shape


# In[20]:


x_train.shape


# In[21]:


x_test.shape


# In[22]:


from sklearn.neural_network import MLPClassifier


# In[23]:


ann= MLPClassifier(hidden_layer_sizes=(100,100,100),random_state=0,max_iter=100, activation='relu')


# In[24]:


ann.fit(x_train, y_train)


# In[25]:


y_pred= ann.predict(x_test)


# In[26]:


from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.metrics import  accuracy_score


# In[27]:


y_test.value_counts()


# In[28]:


ConfusionMatrixDisplay.from_predictions(y_test, y_pred)


# In[29]:


accuracy_score(y_test,y_pred)


# In[30]:


print(classification_report(y_test,y_pred))


# In[ ]:




