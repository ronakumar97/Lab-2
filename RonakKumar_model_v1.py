#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


df = pd.read_csv('breast-cancer-wisconsin.data', sep=",", header=None)


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df[6] = df[6].apply(lambda x: np.nan if x == '?' else x)


# In[6]:


df = df.dropna()
df = df.reset_index(drop=True)


# In[7]:


X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]


# In[8]:


X


# In[9]:


y


# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[12]:


from sklearn.svm import SVC
svm = SVC(C=1, kernel='rbf', cache_size=200)
svm.fit(X_train, y_train)


# In[13]:


pred = svm.predict(X_test)


# In[14]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[15]:


print('Support Vector Classifier' + '\n')
print(classification_report(y_test, pred))
print('\n' + 'Confusion matrix')
print(confusion_matrix(y_test, pred))


# In[16]:


sns.heatmap(confusion_matrix(y_test, pred), annot=True, cmap='Blues')

