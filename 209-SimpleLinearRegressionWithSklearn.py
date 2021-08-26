#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression


# In[25]:


data = pd.read_csv('1.01. Simple linear regression.csv')
data.head() ##visaulize the 5 first data


# In[3]:


## Regression


# In[26]:


x = data['SAT'] ## feature variable - input
y = data['GPA'] ## target variable - output


# In[27]:


x.shape


# In[28]:


y.shape


# In[29]:


##convert into 2d array
x_matrix = x.values.reshape(-1,1)
x_matrix.shape


# In[32]:


reg = LinearRegression()


# In[35]:


reg.fit(x_matrix,y)


# In[38]:


#R-squared
reg.score(x_matrix,y)


# In[40]:


reg.coef_


# In[42]:


reg.intercept_


# In[45]:


#making predictions
reg.predict([[1740]])


# In[48]:


new_data = pd.DataFrame(data=[1740,1760],columns=['SAT'])
new_data


# In[49]:


reg.predict(new_data)


# In[51]:


new_data['Predicted_GPA'] = reg.predict(new_data)
new_data


# In[58]:


plt.scatter(x,y)
yhat = reg.coef_*x_matrix + reg.intercept_
fig = plt.plot(x,yhat, lw=4, c='orange', label='regression line')
plt.xlabel('SAT', fontsize=20)
plt.ylabel('GPA', fontsize=20)
plt.show()


# In[ ]:




