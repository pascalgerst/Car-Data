#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing required packages.
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import os
import warnings 
from scipy import sparse 
get_ipython().run_line_magic('matplotlib', 'inline')

#Import sklearn
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC #Support Vector Classifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier #Neural Network 
#from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
ohe = OneHotEncoder(sparse=False) # Warum spare = false

#Import Lime
import lime
import lime.lime_tabular
from lime import submodular_pick


# In[2]:


#Datenset Bsp
grades = pd.read_csv('C:\\Users\\Lukas\Documents\Python Scripts\student-grades-encoded.csv',sep=',')


# In[3]:


grades.head()


# In[4]:


grades.info() #non-null bezieht sich nur darauf, dass es keine leeren zellen gibt 


# In[5]:


#Preprocessing

bins = (-1,9.5,16.5,20.5)
group_names = ['failed','passed','experts']
grades['Final Grade'] = pd.cut(grades['Final Grade'], bins=bins, labels = group_names)
grades['Final Grade'].unique()


# In[6]:


grades.head(5)


# In[7]:


grades['Final Grade'].value_counts()


# In[8]:


sns.countplot(grades['Final Grade'])


# In[9]:


X = grades.drop('Final Grade', axis = 1) #axis = 1 warum?
X.head()


# In[10]:


X.columns


# In[11]:


Xfeatures = grades.iloc[:,0:25]
Xfeatures.columns
names = Xfeatures.columns
names


# In[12]:


sc = StandardScaler()
X = sc.fit_transform(Xfeatures)
print(X)
X.shape


# In[13]:


X = pd.DataFrame(X, columns = names)
X.head()


# In[14]:


y = grades['Final Grade']
#print(y)
y_label = y.unique()


# In[15]:


y = ohe.fit_transform(grades[['Final Grade']])
print(y)#Wie kann ich nachvollziehen, was was ist


# In[16]:


feature_names = X.columns
class_names = y_label.unique()
print(class_names)


# In[17]:


#Train and Test splitting of data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[18]:


#Applying Standard Scaler -> Wir skalieren Daten, das große nummerische Werte keinen übermäßigen Einfluss nehmen
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)#Hier explizit nur transform, warum?


# #Random Forest Classifier 

# In[19]:


rfc = RandomForestClassifier(n_estimators = 200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
#predict_fn_rf = lambda x: rfc.predict_proba(x).astype(float)


# In[20]:


#How does the model perform
print(classification_report(y_test,pred_rfc)) #Hier werden unsere predictions für y_test also die geschätzten Noten für unsere Final Gread mit den tatsächlichen Noten verglichen

print(sklearn.metrics.accuracy_score(y_test, rfc.predict(X_test)))


# In[21]:


# Erster Testwert als Beispiel
X_test.values[0]
rfc.predict(np.array(X_test.values[0].reshape(1,-1)))
#Beispieltest mit dem Ergebnis 


# In[22]:


# Erster Testwert als Beispiel
X_test.values[10]
rfc.predict(np.array(X_test.values[10].reshape(1,-1)))


# In[23]:


#explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=X.feature_names, class_names=X.target_names, discretize_continuous=True)
explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values,feature_names=feature_names,class_names=class_names,discretize_continuous=True)


# In[29]:


X_test.iloc[0]
#X_test1 = np.array(X_test.values[0].reshape(1,-1))
X_test.values[0].reshape(1,-1)


# In[30]:


exp=explainer.explain_instance(X_test.iloc[0],rfc.predict_proba,num_features=5,top_labels=25)


# In[ ]:


exp.show_in_notebook(show_table=True, show_all=False)


# In[ ]:


#def prob(data):return np.array(list(zip(1-rfc.predict(data),rfc.predict(data))))


# In[ ]:


#explainer = lime.lime_tabular.LimeTabularExplainer(grades[rfc.feature_names()].astype(int).values,  mode='classification',training_labels= X,feature_names=rfc.feature_name())


# In[ ]:


#i = 1
#exp=explainer.explain_instance(grades.loc[i,feat].astype(int).values, prob, num_features=5)


# In[ ]:


#exp.show_in_notebook(show_table=True)


# In[ ]:


#sp_obj = submodular_pick.SubmodularPick(explainer, grades[X.feature_names()].values,\prob, num_features=5, num_exps_desired=10)
#[exp.as_pyplot_figure(label=1) for exp in sp_obj.sp_explanations]


# #SVM Classifier
# 

# In[ ]:


clf = svm.SVC()
clf.fit(X_train, y_train)
pred_clf = clf.predict(X_test)


# In[ ]:


#Lets see how our model performed
print(classification_report(y_test, pred_clf))
print(confusion_matrix(y_test, pred_clf))
print(sklearn.metrics.accuracy_score(y_test, clf.predict(X_test)))


# In[ ]:


explainer = lime.lime_tabular.LimeTabularExplainer(X_train,feature_names=grades,class_names= X,mode='regression')


# In[ ]:


k = np.random.randint(0, X_test.shape[0])
exp.clf = explainer.explain_instance(X_test[k], rfc.predict_proba, num_features=10, top_labels=10)


# In[ ]:


exp.clf.show_in_notebook(show_table=True)


# #Log Regeression 

# In[ ]:


model_logreg = LogisticRegression()
model_logreg.fit(X_train, y_train)


# In[ ]:


model_logreg.predict(X_test)


# In[ ]:


print(sklearn.metrics.accuracy_score(y_test, model_logreg.predict(X_test)))


# In[ ]:





# In[ ]:




