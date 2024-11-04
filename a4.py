#!/usr/bin/env python
# coding: utf-8

# Implement Gradient Descent Algorithm to find the local minima of a function.
# For example, find the local minima of the function y=(x+3)² starting from the point x=2.

# In[7]:


x=2 #starting point
lr=0.01 #learning rate 
precision=0.000001
previous_step_size=1
max_iter=10000
iters=0
gf= lambda x: (x+3)**2


# In[8]:


import matplotlib.pyplot as plt


# In[9]:


gd=[]


# In[10]:


while precision < previous_step_size and iters < max_iter : 
    prev=x
    x= x- lr* gf(prev)
    previous_step_size = abs(x-prev)
    iters+=1
    print('Iteration: ',iters,'Value: ',x) 
    gd.append(x)


# In[11]:


print('Local Minima : ',x)


# In[12]:


plt.plot(gd)

