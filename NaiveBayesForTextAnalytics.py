


import pandas as pd
import numpy as np

train_ds = pd.read_csv("sentiment_train.csv") 


# In[7]:


from sklearn.model_selection import train_test_split


# In[11]:


from sklearn.feature_extraction.text import CountVectorizer
# Initialize the CountVectorizer
count_vectorizer = CountVectorizer()
# Create the dictionary from the corpus
feature_vector = count_vectorizer.fit(train_ds.text)
# Get the feature names
features = feature_vector.get_feature_names()
print("Total number of features: ", len(features))


# In[12]:


train_ds_features = count_vectorizer.transform(train_ds.text) 
type(train_ds_features)


# In[13]:


train_X, test_X, train_y, test_y = train_test_split(train_ds_features,train_ds.sentiment,test_size = 0.3,random_state = 42)


# In[14]:


from sklearn.naive_bayes import BernoulliNB
nb_clf = BernoulliNB()
nb_clf.fit(train_X.toarray(), train_y)


# In[15]:


test_ds_predicted = nb_clf.predict(test_X.toarray())


# In[16]:


from sklearn import metrics
print(metrics.classification_report(test_y, test_ds_predicted))


# In[19]:


import seaborn as sn
from sklearn import metrics
cm = metrics.confusion_matrix(test_y, test_ds_predicted) 
sn.heatmap(cm, annot=True, fmt='.2f');


# In[22]:


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


# In[24]:


from sklearn.feature_extraction import text 
my_stop_words = text.ENGLISH_STOP_WORDS


# In[25]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(analyzer=stemmed_words,max_features = 1000)
feature_vector = tfidf_vectorizer.fit(train_ds.text)
train_ds_features = tfidf_vectorizer.transform(train_ds.text)
features = feature_vector.get_feature_names()


# In[26]:


from sklearn.naive_bayes import GaussianNB
train_X, test_X, train_y, test_y = train_test_split(train_ds_features,
train_ds.sentiment,
test_size = 0.3,
random_state = 42)


# In[27]:


nb_clf = GaussianNB()
nb_clf.fit(train_X.toarray(), train_y)


# In[28]:


test_ds_predicted = nb_clf.predict(test_X.toarray())
print(metrics.classification_report(test_y, test_ds_predicted))


# In[34]:


from nltk.stem import PorterStemmer
# Library for regular expressions
import re
stemmer = PorterStemmer()
def get_stemmed_tokens(doc):
    # Tokenize the documents to words
    all_tokens = [word for word in nltk.word_tokenize(doc)]
    clean_tokens = []
    # Remove all characters other than alphabets. It takes a 
    # regex for matching.
    for each_token in all_tokens:
        if re.search('[a-zA-Z]', each_token):
            clean_tokens.append(each_token)
    # Stem the words
    stemmed_tokens = [stemmer.stem(t) for t in clean_tokens]
    return stemmed_tokens


# In[39]:


import nltk
nltk.download('punkt')
tfidf_vectorizer = TfidfVectorizer(max_features=500,
stop_words='english',
tokenizer=get_stemmed_tokens,
ngram_range=(1,2))
feature_vector = tfidf_vectorizer.fit(train_ds.text)
train_ds_features = tfidf_vectorizer.transform(train_ds.text)
features = feature_vector.get_feature_names()


# In[40]:


train_X, test_X, train_y, test_y = train_test_split(train_ds_features,
train_ds.sentiment,
test_size = 0.3,
random_state = 42)
nb_clf = BernoulliNB()
nb_clf.fit(train_X.toarray(), train_y)
test_ds_predicted = nb_clf.predict(test_X.toarray())
print(metrics.classification_report(test_y, test_ds_predicted))

