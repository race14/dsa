#!/usr/bin/env python
# coding: utf-8

# Implement K-Means clustering/ hierarchical clustering on sales_data_sample.csv dataset. Determine the number of clusters using the elbow method.
# Dataset link : https://www.kaggle.com/datasets/kyanyoga/sample-sales-data

# In[17]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[18]:


df = pd.read_csv('sales_data_sample.csv', encoding='unicode_escape')


# In[19]:


df


# In[20]:


df.info()


# In[21]:


df.describe()


# In[22]:


df.head()


# In[23]:


df.columns


# In[24]:


df.drop([ 'QTR_ID', 'MONTH_ID', 'YEAR_ID',
       'PRODUCTLINE','CUSTOMERNAME', 'PHONE',
       'ADDRESSLINE1', 'ADDRESSLINE2', 'CITY', 'STATE', 'POSTALCODE',
        'TERRITORY', 'CONTACTLASTNAME', 'CONTACTFIRSTNAME','ORDERDATE' ],inplace=True,axis=1)
df


# In[14]:


df.shape


# In[27]:


from sklearn.preprocessing import LabelEncoder, StandardScaler

def convert_categories(col):
    le=LabelEncoder()
    df[col]=le.fit_transform(df[col].values)



# In[28]:


categories=['SALES','STATUS','MSRP','PRODUCTCODE','COUNTRY','DEALSIZE']
for col in categories:
    convert_categories(col)


# In[29]:


df


# In[31]:


stdScaler=StandardScaler()
data= stdScaler.fit_transform(df)


# In[33]:


print(data)


# In[36]:


from sklearn.cluster import KMeans
wcss=[]
for k in range(1,15):
    kmeans=KMeans(n_clusters=k,init='k-means++',random_state=15)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)


# In[37]:


k=list(range(1,15))
plt.plot(k,wcss)
plt.xlabel("Clusters")
plt.ylabel("Scores")
plt.title("Finding right number of clusters")
plt.show()

