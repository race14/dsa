#!/usr/bin/env python
# coding: utf-8

# Predict the price of the Uber ride from a given pickup point to the agreed drop-off location.
# Perform following tasks:
# 1. Pre-process the dataset.
# 2. Identify outliers.
# 3. Check the correlation.
# 4. Implement linear regression and random forest regression models.
# 5. Evaluate the models and compare their respective scores like R2, RMSE, etc.
# Dataset link: https://www.kaggle.com/datasets/yasserh/uber-fares-dataset

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_csv("uber.csv")


# In[3]:


df.shape


# In[4]:


df.columns


# In[5]:


df.dtypes


# In[6]:


df.isnull().sum()


# In[7]:


df.dropna()


# In[8]:


df.describe()


# 1. Pre-process the dataset.
# 

# In[9]:


df['pickup_datetime']=pd.to_datetime(df['pickup_datetime'])
df['pickup_datetime']


# In[10]:


df['pickup_hour']=df['pickup_datetime'].dt.hour
df['pickup_day']=df['pickup_datetime'].dt.dayofweek


# In[11]:


df['pickup_hour']


# In[12]:


df['pickup_day']


# 2. Identify outliers.

# In[13]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[14]:


plt.figure(figsize=(10,6))
sns.boxplot(df['fare_amount'])
plt.title('Fare Amount Outliers')
plt.show()

Q1=df['fare_amount'].quantile(0.25)
Q3=df['fare_amount'].quantile(0.75)
IQR=Q3-Q1
df=df[(df['fare_amount']>= (Q1-1.5*IQR)) & (df['fare_amount']<= (Q3+1.5*IQR)) ]


# In[15]:


plt.figure(figsize=(10,6))
sns.boxplot(df['fare_amount'])
plt.title('Fare Amount Outliers')
plt.show()


# 3. Check the correlation.

# In[16]:


correlation_matrix=df[[ 'fare_amount', 'pickup_datetime',
       'pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
       'dropoff_latitude', 'passenger_count']].corr()
plt.figure(figsize=(10,6))
plt.title("Correlation Matrix")
sns.heatmap(correlation_matrix,annot=True,cmap='coolwarm')
plt.show()


# In[17]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from geopy.distance import geodesic 


df=df[(df['pickup_latitude'].between(-90,90)) &
      (df['pickup_longitude'].between(-180,180)) &
      (df['dropoff_latitude'].between(-90,90)) &
      (df['dropoff_longitude'].between(-90,90)) 
]

df['distance']=df.apply(
    lambda row: geodesic(
        (row['pickup_latitude'],row['pickup_longitude']),
        (row['dropoff_latitude'],row['dropoff_longitude'])).km,axis=1
    
)

x= df[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
       'dropoff_latitude', 'passenger_count','distance']]
y=df['fare_amount']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)




# In[18]:


lr_model=LinearRegression()
lr_model.fit(x_train,y_train)


# In[19]:


rf_model=RandomForestRegressor(n_estimators=100,random_state=42)
rf_model.fit(x_train,y_train)


# In[21]:


from sklearn.metrics import r2_score, mean_squared_error

y_pred_lr=lr_model.predict(x_test)
y_pred_rf=rf_model.predict(x_test)

r2_lr=r2_score(y_test,y_pred_lr)
rmse_lr=np.sqrt(mean_squared_error(y_test,y_pred_lr))

r2_rf=r2_score(y_test,y_pred_rf)
rmse_rf=np.sqrt(mean_squared_error(y_test,y_pred_rf))

print("Linear Regression : R2 Score -> ",r2_lr," and RMSE -> ",rmse_lr)
print("Random Forest Regression : R2 Score -> ",r2_rf," and RMSE -> ",rmse_rf)

