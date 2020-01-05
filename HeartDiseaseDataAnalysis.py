#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as ss
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import confusion_matrix


# In[2]:


varr=pd.read_csv("dataset.csv")


# In[3]:


varr.head(2)


# In[4]:


varr.tail(2)


# In[5]:


varr.shape


# In[ ]:





# In[6]:


varr.describe()


# In[7]:


varr.info()


# In[8]:


varr.columns


# In[9]:


varr.isnull().sum()


# In[10]:


sns.catplot(kind = 'bar', data = varr, y = 'age', x = 'sex', hue = 'target')
plt.title('Distribution of age vs sex with the target class')
plt.show()


# In[11]:


varr.iloc[:2,:-2]


# In[12]:


X=varr.iloc[:, :-1].values
y=varr.iloc[:, -1].values


# In[13]:


X.shape


# In[14]:


y.shape


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 20)


# In[16]:


X_train.shape


# In[17]:


X_test.shape


# In[18]:


y_train.shape


# In[19]:


from sklearn.preprocessing import StandardScaler as ss
sc = ss()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[20]:


regressor = DecisionTreeRegressor()
regressor.fit(X, y)


# In[21]:


import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'


# In[22]:


#visualizing decision tree
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()


# In[ ]:





# In[ ]:





# In[ ]:





# # Model Building

# In[23]:


def main():
    print("Enter Your Choice")
    print("#"*100)
    print("1>Logistic Regression")
    print("2>Decision Tree Classifier")
    ch=int(input())
    if ch==1:
        classifier = LogisticRegression()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        print(y_pred)
        cm_test = confusion_matrix(y_pred, y_test)
        y_pred_train = classifier.predict(X_train)
        cm_train1 = confusion_matrix(y_pred_train, y_train)
        print(cm_train1)
        print()
        print('Accuracy for training set for Logistic Regression = {}'.format((cm_train1[0][0] + cm_train1[1][1])/len(y_train)))
        print('Accuracy for test set for Logistic Regression = {}'.format((cm_test[0][0] + cm_test[1][1])/len(y_test)))
    if ch==2:
        classifier = DecisionTreeClassifier()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        cm_test = confusion_matrix(y_pred, y_test)
        y_pred_train = classifier.predict(X_train)
        export_graphviz(regressor, out_file=dot_data, max_depth=5,filled=True, rounded=True,special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.write_png('Heart Attack.png')
        Image(graph.create_png())
        cm_train = confusion_matrix(y_pred_train, y_train)
        print()
        print('Accuracy for training set for Decision Tree = {}'.format((cm_train[0][0] + cm_train[1][1])/len(y_train)))
        print('Accuracy for test set for Decision Tree = {}'.format((cm_test[0][0] + cm_test[1][1])/len(y_test)))
main()
        


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




