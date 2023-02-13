#!/usr/bin/env python
# coding: utf-8

# # Project 4: Text Classification

# Load all the packages you are going to use.

# **Data manipulation**

# In[2]:


import pandas as pd
import numpy as np
from sklearn import set_config

# to visualize the column transformer and pipeline
set_config(display='diagram')


# **Web scraping**

# In[3]:


import requests


# **Regular Expresssions**

# In[4]:


import re


# **Parsing HTML**

# In[5]:


import os
from bs4 import BeautifulSoup


# **Modeling**

# In[6]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# # 1. Define project goal

# In this project, the goal is to build a text classification model on song lyrics. The task is to predict the artist from a piece of text.
# To train such a model, you first need to collect your own lyrics dataset:
# 
#      Step 1: Download a HTML page with links to songs
# 
#      Step 2: Extract hyperlinks of song pages
# 
#      Step 3: Download and extract the song lyrics
# 
#      Step 4: Vectorize the text using the Bag Of Words method
# 
#      Step 5: Train a classification model that predicts the artist from a piece of text
# 
#      Step 6: Refactor the code into functions
# 
#      Step 7: Write a simple command-line interface for the program
# 
#      Step 8: Upload your code to GitHub
# 
# 

# # 2. Get the data

# ## Web Scraping ł Parsing HTML

# **Get the list of songs urls for a given artist.**

# In[10]:


def get_url_list(url, N):
    
    """ 
    
    Extracts the each song's link by the given url of the artist 
    and returns a url list of N songs.
    
    Parameters:
    ----------
    url: Link of the artist's page
    N: Number of song links to be extracted
    
    """
    
    get_url =requests.get(url) #Get artist url.
    artist = get_url.text #Get a text file from the url.
    artist_soup = BeautifulSoup(artist, "html.parser") #Parsing.
    
    #Loop for the suffixes
    list_of_suffixes = []
    
    
    for song in artist_soup.find_all("td", class_ = "tal qx")[:N]:
        for suffix in song.find_all("a"):
            list_of_suffixes.append((suffix["href"]))
            
            
    #Loop for the names of the artists
    artist = []
    for name in artist_soup.find_all("h1", class_ = "artist"):
        artist.append(name.get_text())
    
    #Loop for constructing the url of the songs
    url_list =[]

    for suffix in list_of_suffixes:
        song_url = "https://www.lyrics.com" + suffix
        url_list.append(song_url)
        
    return url_list #,artist 


# In[11]:


url_list1 = get_url_list("https://www.lyrics.com/artist/La-Femme/2766801", 25)
url_list2 = get_url_list("https://www.lyrics.com/artist/Idles/3252083", 25)
url_list3 = get_url_list ("https://www.lyrics.com/artist/Bj%C3%B6rk/27211", 25)


# In[12]:


url_list2


# **Get the artists' names.**

# In[13]:


def artist_name(url):
    
    """ 
    
    Extracts and returns the name of the artist by the their url.
    
    Parameters:
    ----------
    url: Link of the artist's page
    
    """
    
    get_url =requests.get(url) #Get artist url.
    artist = get_url.text #Get a text file from the url.
    artist_soup = BeautifulSoup(artist, "html.parser") #Parsing.
           
            
    #Loop for the names of the artists
    artist = []
    for name in artist_soup.find_all("h1", class_ = "artist"):
        artist.append(name.get_text())
    
   
    return artist 


# In[14]:


artist1 = artist_name("https://www.lyrics.com/artist/La-Femme/2766801")
artist2 = artist_name("https://www.lyrics.com/artist/Idles/3252083")
artist3 = artist_name("https://www.lyrics.com/artist/Bj%C3%B6rk/27211")


# In[15]:


artists = artist1 + artist2 + artist3


# **Download the lyrics locally.**

# In[19]:


def get_lyrics(artist, url_list):
    
    """
    Downloads and saves the lyrics as text files.
    
    Parameters:
    ----------
    artist: Name of the artist.
    url_list: List of the songs' urls, which was returned from the previous function.
    
    """
    
    #Make a directory for each artist and define its path.
    directory = f"{artist}"
    path = os.path.abspath("")
    full_path = os.path.join(path, directory)
    os.mkdir(full_path)
    
    #Get the titles and lyrics of the songs.
    for url in url_list:
        get_url = requests.get(url)              
        try: #Skip "empty" urls
            song_soup = BeautifulSoup(get_url.text, "html.parser")
            song_title = song_soup.find(class_ = "lyric-title").text

            #for song in song_title:
            with open(f"{full_path}/{song_title}.txt", "w") as file:
                file.write(song_soup.find("pre", id = "lyric-body-text").text)
                   
        except:
            pass
        
        file.close()


# In[20]:


lyrics1 = get_lyrics(artist1, url_list1)
lyrics2 = get_lyrics(artist2, url_list2)
lyrics3 = get_lyrics(artist3, url_list3)


# **Get the list of songs' lyrics for a given artist.**

# **1. From url list.**

# In[21]:


def get_corpus(url_list):
    
    """
    
    Creates a list of lyrics for a given artist's url and returns a corpus, 
    where each string in the list refers to the lyrics of one song.
    
    Parameters:
    ----------
    artist: Name of the artist.

    """
    corpus = []
    
    #Loop for creating a list with all the lyrics of the given url.
    for item in url_list:
        try:
            
            song = requests.get(item)   
            song_soup = BeautifulSoup(song.text, "html.parser")
            lyric = song_soup.find("pre", id = "lyric-body-text").text
            corpus.append(lyric) 
            
        except:
            pass
    return corpus 


# In[22]:


corpus1 = get_corpus(url_list1)
corpus2 = get_corpus(url_list2)
corpus3 = get_corpus(url_list3)


# In[23]:


corpus = corpus1 + corpus2 + corpus3


# In[24]:


corpus1


# **2. From local folders.**

# In[25]:


def get_local_corpus(artist):
    
    """
    
    Creates a list of lyrics for a given artist and returns a corpus, 
    where each string in the list refers to the lyrics of one song.
    
    Parameters:
    ----------
    artist: Name of the artist.

    """
    
    corpus = []
    
    directory = f"{artist}"
    path = os.path.abspath("")
    full_path = os.path.join(path, directory)
    
    #Loop for creating a list with all the lyrics of the given artist.
    for file in os.listdir(directory):
        song = open(f"{full_path}/{file}", "r")
        lyrics = song.read()
        corpus.append(lyrics)
        
    return corpus    


# In[26]:


l_corpus1 = get_local_corpus(artist1)
l_corpus2 = get_local_corpus(artist2)
l_corpus3 = get_local_corpus(artist3)


# In[27]:


l_corpus = l_corpus1 + l_corpus2 + l_corpus3


# In[28]:


l_corpus1


# # 3. Convert text to numerical matrix

# **Get the index for DataFrame.**

# In[29]:


n_songs = [len(corpus1), len(corpus2), len(corpus3)]


# In[30]:


n_songs


# In[31]:


def dictionary(artists, n_songs):

    """
    
    Creates a dictionary of the artists' names and the number of songs, 
    which lyrics are extracted in corpus 
    and returns the labels 
    for the DataFrame's index.
    
    Parameters:
    ----------
    artists: List of artists' names.
    N: List of numbers of songs' lyrics.
    
    """
    
    keys = []
    values = []
    for name in artists:
        keys.append(name)
    for n in n_songs:
        values.append(n)

    dic = {keys[i]: values[i] for i in range(len(keys))}
 
    return dic


# In[32]:


dic = dictionary(artists, n_songs)
dic


# In[33]:


labels = artist1*len(corpus1) + artist2*len(corpus2) + artist3*len(corpus3)


# In[ ]:


#labels = pd.Series(labels)


# **Convert in one step.**

# In[34]:


def vectorize(corpus, labels):
    
    """
    
    Creates a feature matrix with TFIDF values and returns a Dataframe 
    with tokenized and normalized word vectors.
    
    Parameters: 
    ----------
    corpus: List of lyrics, which was returned from the previous function.
    
    """
    
    #Preprocessing.
    for item in corpus:
        item.lower()
        
    #Create vectors.    
    vectorizer = TfidfVectorizer(max_features = 1000, min_df = 2, max_df = 0.5, ngram_range = (1,2), stop_words = "english") 
    vec = vectorizer.fit_transform(corpus)
    
    #Store results to DataFrame.
    df = pd.DataFrame(vec.toarray(), columns= vectorizer.get_feature_names_out(), index = labels)
    
    return df, vectorizer
    #return vec.todense().shape


# In[35]:


df, vectorizer = vectorize(corpus, labels)


# In[36]:


df


# **Convert in two steps, using a pipeline.**

# In[38]:


def vectorize_pipeline(corpus, labels):
    
    """
    
    Creates a feature matrix with TFIDF values and returns a Dataframe 
    with tokenized and normalized word vectors. 
    Two steps of transformation implemented by a pipeline.
    
    Parameters: 
    ----------
    corpus: List of lyrics, which was returned from the previous function.
    
    """
    
    #Preprocessing.
    for item in corpus:
        item.lower()
        
    #Create a pipeline for normalized tf of tf-idf representation.
    pipeline = Pipeline(steps = [
    ("count", CountVectorizer(stop_words='english',ngram_range=(1,2), max_df=0.75)), 
    ("norm", TfidfTransformer())
])
    #Create vectors.    
    vec = pipeline.fit_transform(corpus)
    
    #Store results to DataFrame.
    df = pd.DataFrame(vec.toarray(), columns= pipeline.get_feature_names_out(), index = labels)
    
    return df, pipeline
    #return vec.todense().shape


# In[39]:


df_pip, pipeline = vectorize_pipeline(corpus, labels)


# In[40]:


df_pip


# # 4. Classification models

# ## Prepare data for modeling.

# **Define X and y.**

# In[41]:


X = df.reset_index(drop = True) 
y = df.index #labels


# In[42]:


X.shape, y.shape


# **Split.**

# In[43]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) # apply a train-test split here


# In[44]:


X_train.shape, y_train.shape, X_test.shape, y_test.shape


# ## Modeling.

# In[75]:


def models_evaluation(X_train, X_test, y_train, y_test):
    
    """
    
    Trains several classification models and 
    returns the scores as a DataFrame.
    
    Parameters:
    ----------
    X_train, X_test, y_train, y_test: Train and test datasets 
    to be used for modeling.
    
    """
    
    scores = {}
    models = ["LogisticRegression", "RandomForestClassifier", "MultinomialNB"]
    
    
    for model in models:
        if model == models[0]:
            m = LogisticRegression()
            m.fit(X_train, y_train)
            score_train = m.score(X_train, y_train)
            score_test = m.score(X_test, y_test)
        elif model == models[1]:
            m = RandomForestClassifier(max_depth = 200, n_estimators = 1000)
            m.fit(X_train, y_train)
            score_train = m.score(X_train, y_train)
            score_test = m.score(X_test, y_test)
        elif model == models[2]:
            m = MultinomialNB(alpha = 0.005)
            m.fit(X_train, y_train)
            score_train = m.score(X_train, y_train)
            score_test = m.score(X_test, y_test)

        #else: print("Sorry, this model is not available.")

        scores[f"{model}"] = {
            "train score": score_train,
            "test score": score_test,
        }

    df_scores = pd.DataFrame(scores)
    
    

    return df_scores


# In[76]:


models_scores = models_evaluation(X_train, X_test, y_train, y_test)
models_scores


# **Decide on the model and run it again.**

# # 5. Predict

# In[77]:


def predict(lyrics):
    
    """
    Takes a string as input and returns the prediction on the artist.
    
    Parameters:
    ----------
    text: any string to be used as possible lyric
    
    """
    
    lyrics = [lyrics]
    vec_lyrics = vectorizer.transform(lyrics)
    x_lyrics = vec_lyrics.toarray()
   
    
    m = MultinomialNB(alpha = 0.005)
    m.fit(X, y)
    
    prediction = m.predict(x_lyrics)
    
    probability = m.predict_proba(x_lyrics)
    
    print(f"These lyrics are by {probability.max()*100}% likely to be attributed to: {prediction[0]}")


# In[78]:


X


# In[79]:


predict("Utopia It isn't elsewhere It, it’s here")


# In[ ]:




