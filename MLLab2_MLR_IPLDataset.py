#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

data=pd.read_csv("IPL_Dataset.csv")


data.info()

data.iloc[0:5,15:22] #get 5 rows with 12 columns


# In[ ]:




