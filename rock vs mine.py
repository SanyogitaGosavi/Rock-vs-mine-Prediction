#!/usr/bin/env python
# coding: utf-8

# In[9]:


#importing dependicies
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[10]:


#Data Collection And Data Proessing
#loading Dataset to a pandas data frame
sonar_data = pd.read_csv("C:/Users/gosav/Downloads/sonar data.csv",header = None)


# In[11]:


sonar_data.head()


# In[12]:


#number of rows and columns
sonar_data.shape


# In[13]:


sonar_data.describe()# decirbe stastical measure of the data


# In[14]:


sonar_data[60].value_counts()


# In[15]:


# M --> mine
# R -->rocke


# In[16]:


sonar_data.groupby(60).mean()


# In[17]:


#seprating data and labels
x=sonar_data.drop(columns=60,axis=1)
y=sonar_data [60]


# In[18]:


print(x)
print(y)


# In[19]:


#traning and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1,  stratify=y,random_state=1)


# In[20]:


print(x.shape, x_train.shape,x_test.shape)


# In[ ]:





# In[ ]:





# In[ ]:





# In[21]:


print(x_train)
print(y_train)


# In[22]:


#model traning
model = LogisticRegression()


# In[24]:


#train the LogisticRegression model with training data
model.fit(x_train,y_train)


# In[25]:


#accuracy training data
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction,y_train)


# In[26]:


print("Accuracy on training data:",training_data_accuracy)


# In[27]:


x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction,y_test)


# In[28]:


print("Accuracy on test data:",test_data_accuracy)


# In[29]:


import numpy as np


# In[31]:


#Making prediction system

input_data=(0.02,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.066,0.2273,0.31,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.555,0.6711,0.6415,0.7104,0.808,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.051,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.018,0.0084,0.009,0.0032,
)
input_data_as_numpy_array=np.asarray(input_data)
#reshape thr np array as we predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)
if(prediction[0]=='R'):
    print("The object is rock")
else:
    print("The object is a mine")


# In[ ]:




