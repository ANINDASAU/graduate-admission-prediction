#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# In[2]:


df = pd.read_csv("/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv")

# In[3]:


df.head()

# In[4]:


df.shape

# In[5]:


df.info()

# In[6]:


df.duplicated().sum()

# In[8]:


df.drop(columns=['Serial No.'],inplace=True)

# In[9]:


df.head()

# In[10]:


X = df.iloc[:,0:-1]
y = df.iloc[:,-1]

# In[11]:


X

# In[12]:


y

# In[13]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

# In[14]:


X_train

# In[15]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# In[16]:


X_train_scaled

# In[17]:


import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense

# In[24]:


model = Sequential()

model.add(Dense(7,activation='relu',input_dim=7))
model.add(Dense(7,activation='relu'))
model.add(Dense(1,activation='linear'))

# In[25]:


model.summary()

# In[26]:


model.compile(loss='mean_squared_error', optimizer ='Adam')

# In[27]:


history= model.fit(X_train_scaled,y_train,epochs=100,validation_split=0.2)

# In[28]:


y_pred = model.predict(X_test_scaled)

# In[29]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)

# In[30]:


import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

# In[ ]:



