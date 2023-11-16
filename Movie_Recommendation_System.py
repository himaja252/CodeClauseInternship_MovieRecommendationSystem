#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import ast


# In[2]:


movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


movies.head()


# In[4]:


credits.head()


# In[5]:


movies=movies.merge(credits,on='title')


# In[6]:


movies.head()


# In[7]:


movies.info()


# In[8]:


movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[9]:


movies.head()


# In[10]:


movies.isnull().sum()


# In[11]:


movies.dropna(inplace=True)


# In[12]:


movies.isnull().sum()


# In[13]:


movies.duplicated().sum()


# In[14]:


def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[15]:


movies['genres']=movies['genres'].apply(convert)


# In[16]:


movies['keywords']=movies['keywords'].apply(convert)


# In[17]:


def convert3(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter!=3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L


# In[18]:


movies['cast']=movies['cast'].apply(convert3)


# In[19]:


def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
            break
    return L


# In[20]:


movies['crew']=movies['crew'].apply(fetch_director)


# In[21]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[22]:


movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])
movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])


# In[23]:


movies.head()


# In[24]:


movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[25]:


movies.head()


# In[26]:


new_dataframe=movies[['movie_id','title','tags']]


# In[27]:


new_dataframe['tags']=new_dataframe['tags'].apply(lambda x:" ".join(x))


# In[28]:


new_dataframe.head()


# In[29]:


new_dataframe['tags']=new_dataframe['tags'].apply(lambda x:x.lower())


# In[30]:


import nltk 


# In[31]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)


# In[32]:


new_dataframe['tags']=new_dataframe['tags'].apply(stem)


# In[33]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')


# In[34]:


vectors=cv.fit_transform(new_dataframe['tags']).toarray()


# In[35]:


vectors


# In[36]:


from sklearn.metrics.pairwise import cosine_similarity


# In[37]:


cosine_similarity(vectors)


# In[38]:


similarity=cosine_similarity(vectors)


# In[39]:


def recommend(movie):
    movie_index=new_dataframe[new_dataframe['title']==movie].index[0]
    distances=similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_dataframe.iloc[i[0]].title)


# In[40]:


recommend('Avatar')


# In[45]:


recommend('Aliens')


# In[ ]:




