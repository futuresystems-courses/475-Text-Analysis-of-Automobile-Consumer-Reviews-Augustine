"""
================================================================================
Final Project : INFO-I590-34717
================================================================================
Author: Augustine Joseph
Starter Code: written by Olivier Grisel, Lars Buitinck
License : BSD 3 Clause
                                                                 
        This program applies Non-negative Matrix Factorization                                    
 	on a corpus of documents and extract additive models of 
 	the topic structure of the corpus. These documents are
 	extracted from car reviews at edmunds.com. This program
 	can extract top 6 favorite features and top 6 most disliked
 	features of a car from a large number of review sets.  
 								 
 	Inputs		:  300 sample car reviews from edmunds.com     
                            	                 
 	Output		:   Writes 6 most popular features of a car
 	                    Writes 6 most disliked features of a car
 	                     	 
 	Library         :   NLTK 3.0, scikit-learn 0.17				 
 						
================================================================================

"""

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.decomposition import NMF
from nltk.stem import WordNetLemmatizer
import json
import requests
#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt

from pandas.io.json import json_normalize
 
 
 
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))

    
    
"""
Function
--------
extract_topic:

This function will extract the most popular topic of discussion from text data

Inputs
-------
topic_list : List object that contains discussions in a text format
n_samples  : Number of samples
n_features : Number of features
n_topics   : Number of topics
n_top_words: Number of top words
   
"""   
def extract_topic(topic_list, n_samples, n_features, n_topics, n_top_words):

    # Define tokenizers and preprocessors
    base_tokenizer = CountVectorizer().build_tokenizer()
    lemmatizer = WordNetLemmatizer()
   
    tokenize = None
    tokenize = lambda text : (lemmatizer.lemmatize(item) for item in base_tokenizer(text))


    # Using tf-idf features for NMF.
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=0.001, #max_features=n_features, 
                             stop_words='english', ngram_range=(1,2), tokenizer = tokenize)
    tfidf = vectorizer.fit_transform(topic_list[:n_samples]) 
    
    # Fitting the NMF model
    
    nmf = NMF(n_components=n_topics, random_state=1).fit(tfidf)
   
    tfidf_feature_names = vectorizer.get_feature_names()
    print_top_words(nmf, tfidf_feature_names, n_top_words)
    
# Studying reviews on 2008 hond accord from edmunds.com
# Extract 6 Most favorite features identified by the reviewers   
# Extract 6 Most disliked features identified by the reviewers    
             
n_samples = 300
n_features = 23
n_topics = 6
n_top_words = 6

# Collect data from edmunds.com through API calls
        
url ='https://api.edmunds.com/api/vehiclereviews/v2/honda/accord/2008?sortby=thumbsUp%3AASC&pagenum=1&pagesize=768&fmt=json&api_key='
    

data = requests.get(url).text
data = json.loads(data)  # load a json string into a collection of lists and dicts

dataset = json_normalize(data['reviews'])

imp = dataset['suggestedImprovements']
fav = dataset['favoriteFeatures']

si_list = imp.tolist()
#si_list = filter(None, si_list)
#print(len(si_list))

fav_list = fav.tolist()
#fav_list = filter(None, fav_list)
#print(type(fav_list))

"""
with open("./the_filename", 'w') as f:
   for s in fav_list:
       f.write(s + '\n')

"""

print
print
print("Most Popular Features of this Car")
print("=================================")

extract_topic(fav_list, n_samples, n_features, n_topics, n_top_words)

print
print
print("Most Disliked Features of this Car")
print("=================================")

extract_topic(si_list, n_samples , n_features, n_topics, n_top_words)


