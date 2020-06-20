#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing required packages.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
#from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


cd E:\Uni Frankfurt\Seminar Informationssysteme\Datasets


# In[3]:


data = pd.read_csv('car data.csv',sep=',')


# In[4]:


data.head()


# In[5]:


data.info()


# In[6]:


data.isnull().sum()


# In[7]:


#Preprocessing
bins = (0, 8, 16, 25, np.inf)
group_names = ['junk', 'low class', 'middle class', 'high class']
data['Selling_Price'] = pd.cut(data['Selling_Price'], bins = bins, labels = group_names)
data['Selling_Price'].unique


# In[8]:


label_Selling_Price = LabelEncoder()


# In[9]:


data['Selling_Price'] = label_Selling_Price.fit_transform(data['Selling_Price'])


# In[10]:


data.head()


# In[11]:


data['Selling_Price'].value_counts()


# In[12]:


sns.countplot(data['Selling_Price'])


# In[13]:


#Now separate the dataset as response variable and feature variables
X = data.drop('Selling_Price', axis = 'columns')
y = data['Selling_Price']


# In[14]:


y.shape


# In[15]:


X.head()


# In[16]:


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)


# In[17]:


from sklearn.compose import make_column_transformer


# In[18]:


ohe.fit_transform(data[['Car_Name']])


# In[19]:


ohe.fit_transform(data[['Year']])


# In[20]:


ohe.fit_transform(data[['Fuel_Type']])


# In[21]:


ohe.fit_transform(data[['Seller_Type']])


# In[22]:


ohe.fit_transform(data[['Transmission']])


# In[23]:


ohe.categories_


# In[24]:


X = ohe.fit_transform(X)


# In[25]:


X.head()


# In[ ]:


X.info()


# In[27]:


#Train and Test splitting of data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42) # randomstate not too important 


# In[28]:


# Applying Standard scaling to get optimized result

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[29]:


print(X)


# In[ ]:





# In[ ]:


column_trans = make_column_transformer((OneHotEncoder(), ['Car_Name', 'Year', 'Fuel_Type', 'Seller_Type', 'Transmission']),remainder = 'passthrough')


# In[ ]:


X = column_trans.fit_transform(X)

