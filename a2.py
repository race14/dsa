#!/usr/bin/env python
# coding: utf-8

#     Classify the email using the binary classification method. Email Spam detection has two 
#     states: a) Normal State – Not Spam, b) Abnormal State – Spam. Use K-Nearest Neighbors and 
#     Support Vector Machine for classification. Analyze their performance. 
#     Dataset link: The emails.csv dataset on the Kaggle 
#     https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv 

# In[262]:


import pandas as pd
import matplotlib.pyplot as plt


# In[263]:


df= pd.read_csv('emails.csv')


# In[264]:


df.shape


# In[265]:


df.columns


# In[266]:


df.head()


# In[267]:


df.info


# In[268]:


df.describe()


# In[269]:


df.isnull().sum()


# In[270]:


#input data
x= df.drop(['Email No.','Prediction'],axis=1)

#Output Data
y= df['Prediction']
  


# In[271]:


import seaborn as sns


# In[272]:


sns.countplot(x=y)


# In[273]:


#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)


# In[274]:


x_scaled


# In[275]:


#Cross-Validation
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(
    x_scaled,y,random_state=0,test_size=0.25
)


# KNN

# In[276]:


#import the class 
from sklearn.neighbors import KNeighborsClassifier


# In[277]:


#Create the Object 
knn = KNeighborsClassifier(n_neighbors=10)


# In[278]:


#Train the algorithm
knn.fit(x_train,y_train)


# In[279]:


#Predict on test data
y_pred=knn.predict(x_test)


# In[280]:


#Import the evaluation matrix
from sklearn.metrics import ( ConfusionMatrixDisplay,
    accuracy_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    classification_report)


# In[281]:


ConfusionMatrixDisplay.from_predictions(y_test,y_pred)


# In[282]:


# Plot Precision-Recall Curve
PrecisionRecallDisplay.from_estimator(knn, x_test, y_test)
plt.title("KNN - Precision-Recall Curve")
plt.show()

# Plot ROC Curve
RocCurveDisplay.from_estimator(knn, x_test, y_test)
plt.title("KNN - ROC Curve")
plt.show()


# In[283]:


accuracy_score(y_test,y_pred)


# In[284]:


print(classification_report(y_test,y_pred))


# SVM

# In[285]:


from sklearn.svm import SVC


# In[286]:


svm = SVC(kernel='linear')


# In[287]:


svm.fit(x_train,y_train)


# In[288]:


y_pred=svm.predict(x_test)


# In[289]:


ConfusionMatrixDisplay.from_predictions(y_test,y_pred)


# In[290]:


# Plot Precision-Recall Curve
PrecisionRecallDisplay.from_estimator(svm, x_test, y_test)
plt.title("SVM - Precision-Recall Curve")
plt.show()

# Plot ROC Curve
RocCurveDisplay.from_estimator(svm, x_test, y_test)
plt.title("SVM - ROC Curve")
plt.show()


# In[291]:


accuracy_score(y_test,y_pred)


# In[292]:


print(classification_report(y_test,y_pred))

