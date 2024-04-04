#!/usr/bin/env python
# coding: utf-8

# # KFOLD CROSS VALIDATION

# In[16]:


from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np


# In[17]:


iris = load_iris()
X,y = iris.data, iris.target


# In[18]:


kf = KFold(n_splits=5, shuffle=True, random_state=42)


# In[19]:


knn = KNeighborsClassifier(n_neighbors=3)


# In[20]:


scores = []


# In[21]:


for train_index, test_index in kf.split(X):
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    knn.fit(x_train,y_train)
    y_pred = knn.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    scores.append(score)


# In[22]:


mean_accuracy = np.mean(scores)


# In[23]:


print("K fold cross validation",scores)
print("ortalama hata skoru", mean_score)


# # Confusion Matrix

# In[25]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix


# In[26]:


iris = load_iris()
x_train, x_test, y_train, y_test= train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)


# In[27]:


clf = DecisionTreeClassifier()
clf.fit(x_train,y_train)


# In[28]:


y_pred = clf.predict(x_test)


# In[29]:


cm = confusion_matrix(y_test,y_pred)


# In[30]:


labels = ["Setosa", "Versicolor", "Virginica"]
df_cm= pd.DataFrame(cm, index=labels, columns=labels)


# In[31]:


plt.figure(figsize=(8,6))
sns.heatmap (df_cm, annot=True, cmap=plt.cm.Blues)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[ ]:




