#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
get_ipython().run_line_magic('matplotlib', 'inline')

beer_df=pd.read_csv('beer.csv')
beer_df


# In[3]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaled_beer_df=scaler.fit_transform(beer_df[['calories', 'sodium', 'alcohol', 'cost']])


# In[4]:


scaled_beer_df[:]


# In[5]:


cmap=sn.cubehelix_palette(as_cmap=True, rot=-.3, light=1)
sn.clustermap(scaled_beer_df, cmap=cmap, linewidths=.2, figsize=(8,8))


# In[6]:


beer_df.iloc[[10,16]]


# In[7]:


beer_df.iloc[[2,18]]


# In[9]:


from sklearn.cluster import KMeans

cluster_range=range(1,10)
cluster_errors=[]
for num_clusters in cluster_range:
    clusters=KMeans(num_clusters)
    clusters.fit(scaled_beer_df)
    cluster_errors.append(clusters.inertia_)
plt.figure(figsize=(6, 4))
plt.plot(cluster_range, cluster_errors, marker="o")


# In[10]:


scaler=StandardScaler()
scaled_beer_df=scaler.fit_transform(beer_df[['calories', 'sodium', 'alcohol', 'cost']])


# In[11]:


k=3
clusters=KMeans(k, random_state=42)
clusters.fit(scaled_beer_df)
beer_df["clusterid"]=clusters.labels_


# In[12]:


beer_df[beer_df.clusterid==0]


# In[13]:


beer_df[beer_df.clusterid==1]


# In[14]:


beer_df[beer_df.clusterid==2]


# In[15]:


#Hierachical clustering
from sklearn.cluster import AgglomerativeClustering

h_clusters=AgglomerativeClustering(3)
h_clusters.fit(scaled_beer_df)
beer_df["h_clusterid"]=h_clusters.labels_


# In[16]:


beer_df[beer_df.clusterid==0]


# In[17]:


beer_df[beer_df.clusterid==1]


# In[18]:


beer_df[beer_df.clusterid==2]

