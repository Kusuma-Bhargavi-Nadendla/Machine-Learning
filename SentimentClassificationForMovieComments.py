#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np
import warnings
warnings.filterwarnings("ignore")
train_ds = pd.read_csv("sentiment_train.csv") 
train_ds.head(5)


# In[3]:


pd.set_option("max_colwidth", 800)
train_ds[train_ds.sentiment == 1][0:5]


# In[4]:


train_ds[train_ds.sentiment == 0][0:5]


# In[5]:


train_ds.info()


# In[6]:


import matplotlib.pyplot as plt 
import seaborn as sn
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(6,5))
# Create count plot
ax = sn.countplot(x="sentiment", data=train_ds)
# Annotate
for p in ax.patches:
    ax.annotate(p.get_height(), (p.get_x()+0.1, p.get_height()+50))


# In[8]:


from sklearn.feature_extraction.text import CountVectorizer
# Initialize the CountVectorizer
count_vectorizer = CountVectorizer()
# Create the dictionary from the corpus
feature_vector = count_vectorizer.fit(train_ds.text)
# Get the feature names
features = feature_vector.get_feature_names()
print("Total number of features: ", len(features))


# In[9]:


import random
random.sample(features, 10)


# In[10]:


train_ds_features = count_vectorizer.transform(train_ds.text) 
type(train_ds_features)


# In[11]:


train_ds_features.shape


# In[12]:


train_ds_features.getnnz()


# In[18]:


print('Density of the matrix: ', train_ds_features.getnnz() * 100 /(train_ds_features.shape[0] * train_ds_features.shape[1]))


# In[19]:


# Converting the matrix to a dataframe
train_ds_df = pd.DataFrame(train_ds_features.todense()) 
# Setting the column names to the features i.e. words 
train_ds_df.columns = features


# In[20]:


train_ds[0:1]


# In[21]:


train_ds_df.iloc[0:1, 150:157]


# In[23]:


train_ds_df[['the', 'da', 'vinci', 'code', 'book', 'is', 'just', 'awesome']][0:1]


# In[24]:


# Summing up the occurrences of features column wise 
features_counts = np.sum(train_ds_features.toarray(), axis = 0) 
feature_counts_df = pd.DataFrame(dict(features = features,
counts = features_counts))


# In[26]:


plt.figure(figsize=(12,5))
plt.hist(feature_counts_df.counts, bins=50, range = (0, 2000)); 
plt.xlabel("Frequency of words")
plt.ylabel("Density");


# In[27]:


len(feature_counts_df[feature_counts_df.counts == 1])


# In[28]:


# Initialize the CountVectorizer
count_vectorizer = CountVectorizer(max_features=1000) 
# Create the dictionary from the corpus
feature_vector = count_vectorizer.fit(train_ds.text)
# Get the feature names
features = feature_vector.get_feature_names()
# Transform the document into vectors
train_ds_features = count_vectorizer.transform(train_ds.text)
# Count the frequency of the features
features_counts = np.sum(train_ds_features.toarray(), axis = 0) 
feature_counts = pd.DataFrame(dict(features = features, 
counts = features_counts))


# In[31]:


feature_counts.sort_values('counts', ascending = False)[0:15]


# In[36]:


from sklearn.feature_extraction import text 
my_stop_words = text.ENGLISH_STOP_WORDS
#Printing first few stop words
print('Few stop words: ', list(my_stop_words)[0:10])


# In[38]:


# Adding custom words to the list of stop words
my_stop_words = text.ENGLISH_STOP_WORDS.union(['harry', 'potter', 
'code', 'vinci', 'da','harry', 'mountain', 'movie', 'movies'])


# In[39]:


# Setting stop words list
count_vectorizer = CountVectorizer(stop_words = my_stop_words,max_features = 1000)
feature_vector = count_vectorizer.fit(train_ds.text)


# In[40]:


train_ds_features = count_vectorizer.transform(train_ds.text)
features = feature_vector.get_feature_names()
features_counts = np.sum(train_ds_features.toarray(), axis = 0) 
feature_counts = pd.DataFrame(dict(features = features,
counts = features_counts))


# In[42]:


feature_counts.sort_values('counts', ascending = False)[0:15]


# In[50]:


from nltk.stem.snowball import PorterStemmer
stemmer = PorterStemmer()
analyzer = CountVectorizer().build_analyzer()
#Custom function for stemming and stop word removal 
def stemmed_words(doc):
    ### Stemming of words
    stemmed_words = [stemmer.stem(w) for w in analyzer(doc)]
    ### Remove the words in stop words list
    non_stop_words = [word for word in stemmed_words if word not in my_stop_words]
    return non_stop_words


# In[54]:


count_vectorizer = CountVectorizer(analyzer=stemmed_words,max_features = 1000)
feature_vector = count_vectorizer.fit(train_ds.text)
train_ds_features = count_vectorizer.transform(train_ds.text)
features = feature_vector.get_feature_names()
features_counts = np.sum(train_ds_features.toarray(), axis = 0) 
feature_counts = pd.DataFrame(dict(features = features,counts = features_counts))
feature_counts.sort_values('counts', ascending = False)[0:15]


# In[56]:


# Convert the document vector matrix into dataframe
train_ds_df = pd.DataFrame(train_ds_features.todense())
# Assign the features names to the column
train_ds_df.columns = features
# Assign the sentiment labels to the train_ds
train_ds_df['sentiment'] = train_ds.sentiment


# In[57]:


sn.barplot(x = 'sentiment', y = 'awesom', data = train_ds_df, estimator=sum);


# In[59]:


sn.barplot(x = 'sentiment', y = 'realli', data = train_ds_df, estimator=sum);


# In[60]:


sn.barplot(x = 'sentiment', y = 'hate', data = train_ds_df, estimator=sum);

