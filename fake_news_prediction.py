#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv('Downloads/train.csv')


# In[3]:


df.head()


# In[9]:


df.shape


# In[10]:


df.isnull().sum()


# In[11]:


df.dropna(inplace=True)


# In[12]:


df.shape


# In[13]:


df.dtypes


# In[14]:


df['label']=df['label'].astype(str)


# In[15]:


df.dtypes


# In[16]:


import seaborn as sns
def create_distribution(feature):
    return sns.countplot(df[feature])


# In[17]:


create_distribution('label')


#  

# In[18]:


msg=df.copy()


# In[19]:


msg.reset_index(inplace=True)


# In[20]:


msg.head(20)


# In[21]:


msg.drop(['index','id'],axis=1,inplace=True)


# In[22]:


msg.head(10)


# In[27]:


import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
ps=PorterStemmer()


# In[31]:


corpus=[]
sentence=[]
for i in range(0,len(msg)):
    review=re.sub('[^a-zA-Z]',' ',msg['title'][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    sentence=' '.join(review)
    corpus.append(sentence)


# In[34]:


len(corpus)


# In[35]:


from sklearn.feature_extraction.text import CountVectorizer


# In[36]:


cv=CountVectorizer(max_features=500,ngram_range=(1,3))


# In[37]:


X=cv.fit_transform(corpus).toarray()


# In[38]:


X


# In[40]:


cv.get_feature_names()[0:20]


# In[41]:


y=msg['label']


# In[42]:


from sklearn.model_selection import train_test_split


# In[43]:


X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.25, random_state=42)


# In[44]:


X_test


# In[45]:


from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()


# In[46]:


classifier.fit(X_train,y_train)


# In[48]:


pred=classifier.predict(X_test)
pred


# In[49]:


from sklearn.metrics import confusion_matrix,accuracy_score
accuracy_score(y_test,pred)


# In[50]:


confusion_matrix(y_test,pred)


# In[52]:


from sklearn.linear_model import PassiveAggressiveClassifier


# In[53]:


linear_clf=PassiveAggressiveClassifier()


# In[54]:


linear_clf.fit(X_train,y_train)


# In[55]:


prediction=linear_clf.predict(X_test)


# In[56]:


accuracy_score(y_test,prediction)


# In[57]:


confusion_matrix(y_test,prediction)

