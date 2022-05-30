#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[ ]:


X_train = pd.read_csv('data/X_train.csv')
y_train = pd.read_csv('data/y_train.csv')
X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv')


# In[ ]:


model = LogisticRegression()
model.fit(X_train, y_train['gender'])


# In[ ]:


y_predicted = model.predict(X_test)
acc = accuracy_score(y_test, y_predicted)
with open('build_accuracy.txt', 'w') as file:
    file.write(str(acc))
    file.write('\n')


with open('build_accuracy.txt') as file:
    acc = [float(line.rstrip()) for line in file]
    
builds = list(range(1, len(acc) + 1))

plt.xlabel('build')
plt.ylabel('accuracy')
plt.plot(builds, acc, 'ro')
plt.show()
plt.savefig('builds_accuracy.jpg')
