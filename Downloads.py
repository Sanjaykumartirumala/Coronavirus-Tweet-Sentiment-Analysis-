#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# In[22]:


df = pd.read_excel('Credit Card Defaulter Prediction.xlsx')


# In[25]:


label_encoders = {}
categorical_columns = ['SEX', 'EDUCATION', 'MARRIAGE', 'default ']

for column in categorical_columns:
    # Convert all values to string type before encoding
    df[column] = df[column].astype(str)
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])


# In[26]:


X = df.drop('default ', axis=1)
y = df['default ']


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[28]:


model = LogisticRegression()
model.fit(X_train, y_train)


# In[29]:


y_pred_prob = model.predict_proba(X_test)[:, 1]


# In[30]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)


# In[31]:


plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[ ]:




