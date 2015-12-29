"""
================================================================================
Final Project : INFO-I590-34717
================================================================================
Author: Augustine Joseph
Starter Code: written by Olivier Grisel, Lars Buitinck
              Code from Rotten Tomatoes Movie Review Data Analysis
License : BSD 3 Clause
                                                                 
        This program is an application of Naive Bayes classifier 
        to build a prediction model for whether a review is 
        postive or negative(favorite car or not so favorite car), 
        depending on the text of the car reviews at edmunds.com. 
  	 
 								 
 	Inputs		:  300 sample car reviews from edmunds.com     
                            	                 
 	Output		:   Predict if a car review is positive or
 	                    negative. Present the accuracy of the 
 	                    prediction model.
 	                   
 	Library         :   NLTK 3.0, scikit-learn 0.17				 
 						
================================================================================

"""



from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             f1_score)
from sklearn.cross_validation import train_test_split 
from sklearn.calibration import CalibratedClassifierCV
from sklearn import naive_bayes  
from sklearn.metrics import roc_auc_score  
from nltk.corpus import stopwords  
import json
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.io.json import json_normalize


 

"""
Function
--------
calibration_plot

Builds a plot like the one above, from a classifier and review data

Inputs
-------
clf : Classifier object
    A MultinomialNB classifier
X : (Nexample, Nfeature) array
    The bag-of-words data
Y : (Nexample) integer array
    1 if a review is Fresh
"""     


def calibration_plot(clf, xtest, ytest):
    prob = clf.predict_proba(xtest)[:, 1]
    outcome = ytest
    data = pd.DataFrame(dict(prob=prob, outcome=outcome))
    #data = pd.DataFrame(dict(outcome=outcome, prob=prob ))
    #group outcomes into bins of similar probability
    bins = np.linspace(0, 1, 20)
    cuts = pd.cut(prob, bins)
    binwidth = bins[1] - bins[0]
    
    #freshness ratio and number of examples in each bin
    cal = data.groupby(cuts).outcome.agg(['mean', 'count'])
    cal['pmid'] = (bins[:-1] + bins[1:]) / 2
    cal['sig'] = np.sqrt(cal.pmid * (1 - cal.pmid) / cal['count'])
        
    #the calibration plot
    ax = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    p = plt.errorbar(cal.pmid, cal['mean'], cal['sig'])
    plt.plot(cal.pmid, cal.pmid, linestyle='--', lw=1, color='k')
    plt.ylabel("Empirical Favorite")
    remove_border(ax)
  
    #the distribution of Favorite
    ax = plt.subplot2grid((3, 1), (2, 0), sharex=ax)
    
    plt.bar(left=cal.pmid - binwidth / 2, height=cal['count'],
            width=.95 * (bins[1] - bins[0]),
            fc=p[0].get_color())
    
    plt.xlabel("Predicted Favorite")
    remove_border()
    plt.ylabel("Number")

"""
  Ploting graphs     
                

"""  
                                        
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 30)

# set some nicer defaults for matplotlib
from matplotlib import rcParams

#these colors come from colorbrewer2.org. Each is an RGB triplet
dark2_colors = [(0.10588235294117647, 0.6196078431372549, 0.4666666666666667),
                (0.8509803921568627, 0.37254901960784315, 0.00784313725490196),
                (0.4588235294117647, 0.4392156862745098, 0.7019607843137254),
                (0.9058823529411765, 0.1607843137254902, 0.5411764705882353),
                (0.4, 0.6509803921568628, 0.11764705882352941),
                (0.9019607843137255, 0.6705882352941176, 0.00784313725490196),
                (0.6509803921568628, 0.4627450980392157, 0.11372549019607843),
                (0.4, 0.4, 0.4)]

rcParams['figure.figsize'] = (10, 6)
rcParams['figure.dpi'] = 150
rcParams['axes.color_cycle'] = dark2_colors
rcParams['lines.linewidth'] = 2
rcParams['axes.grid'] = False
rcParams['axes.facecolor'] = 'white'
rcParams['font.size'] = 14
rcParams['patch.edgecolor'] = 'none'


def remove_border(axes=None, top=False, right=False, left=True, bottom=True):
    
    #Minimize chartjunk by stripping out unnecesary plot borders and axis ticks
    
    #The top/right/left/bottom keywords toggle whether the corresponding plot border is drawn
 
    ax = axes or plt.gca()
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)
    
    #turn off all ticks
    ax.yaxis.set_ticks_position('none')#
    ax.xaxis.set_ticks_position('none')
    
    #now re-enable visibles
    if top:
        ax.xaxis.tick_top()
    if bottom:
        ax.xaxis.tick_bottom()
    if left:
        ax.yaxis.tick_left()
    if right:
        ax.yaxis.tick_right()
                                                                                                                    
            
             
###################### 

"""                      
clf : Classifier object
    A MultinomialNB classifier
X : (Nexample, Nfeature) array
    The bag-of-words data generated from the review text
Y : (Nexample) integer array
    1 if average rating is 4 or above indicating favorite car
    0 if average rating is below 4 indicating not so favorite car
"""  
  
# Collect data from edmunds.com through API calls
    
url ='https://api.edmunds.com/api/vehiclereviews/v2/honda/accord/2008?sortby=thumbsUp%3AASC&pagenum=1&pagesize=768&fmt=json&api_key='
 
data = requests.get(url).text
data = json.loads(data)  # load a json string into a collection of lists and dicts

dataset = json_normalize(data['reviews'])

rating = dataset['averageRating']
rating_list = rating.tolist()

reviews = dataset['text']
reviews_list = reviews.tolist()

#print(len(rating_list))

# 
stopset = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset)


# Generating the Y array

Y = np.asarray(rating_list)
   
Y=Y.astype(np.float)
Y=Y.astype(np.int)
Y[Y < 4] = int(0) # Average rating below 4 is not considered as favorite
Y[Y != 0] = int(1) # Average rating of 4 or above is not considered as favorite


# Generating the X array

X = vectorizer.fit_transform(reviews_list)
X = X.toarray()

#print Y.shape
#print X.shape

sample_weight = np.random.RandomState(3).rand(Y.shape[0])

X_train, X_test, Y_train, Y_test, sw_train, sw_test = train_test_split(X, Y, sample_weight, test_size=0.9, random_state=3)

 # Fitting the Naive Bayes model

clf = naive_bayes.MultinomialNB()
clf.fit(X_train, Y_train) 

#print "Accuracy: %0.2f%%" % (100 * clf.score(X_test, Y_test))


training_accuracy = clf.score(X_train, Y_train)
test_accuracy = clf.score(X_test, Y_test)

prob_pos_clf = clf.predict_proba(X_test)[:, 1]
Y_pred = clf.predict(X_test)
roc = roc_auc_score(Y_test, prob_pos_clf)


# Naive-Bayes with sigmoid calibration
clf_sigmoid = CalibratedClassifierCV(clf, cv=2, method='sigmoid')
clf_sigmoid.fit(X_train, Y_train, sw_train)
prob_pos_sigmoid = clf_sigmoid.predict_proba(X_test)[:, 1]
Y_pred_sigmoid = clf_sigmoid.predict(X_test)
roc_sigmoid = roc_auc_score(Y_test, prob_pos_sigmoid)

clf_score = brier_score_loss(Y_test, prob_pos_clf, sw_test)
#print("No calibration: %1.3f" % clf_score)

clf_sigmoid_score = brier_score_loss(Y_test, prob_pos_sigmoid, sw_test)
#print("With sigmoid calibration: %1.3f" % clf_sigmoid_score)

print("Naive Bayes:")
print("\tROC: %1.3f" % (roc))
print("\tBrier: %1.3f" % (clf_score))
print("\tPrecision: %1.3f" % precision_score(Y_test, Y_pred))
print("\tRecall: %1.3f" % recall_score(Y_test, Y_pred))
print("\tF1: %1.3f\n" % f1_score(Y_test, Y_pred))

print("Naive Bayes + sigmoid:")
print("\tROC: %1.3f" % (roc_sigmoid))
print("\tBrier: %1.3f" % (clf_sigmoid_score))
print("\tPrecision: %1.3f" % precision_score(Y_test, Y_pred_sigmoid))
print("\tRecall: %1.3f" % recall_score(Y_test, Y_pred_sigmoid))
print("\tF1: %1.3f\n" % f1_score(Y_test, Y_pred_sigmoid))


print "Accuracy on training data: %0.2f" % (training_accuracy)
print "Accuracy on test data:     %0.2f" % (test_accuracy)

# Plot the predicted probabilities

calibration_plot(clf_sigmoid, X_test, Y_test)
remove_border()
plt.show()


plt.figure()

order = np.lexsort((prob_pos_clf, ))
plt.plot(prob_pos_clf[order], 'r', label='No calibration (%1.3f)' % clf_score)
plt.plot(prob_pos_sigmoid[order], #'b', linewidth=3,
         linestyle='-', lw=1, color='k',
         label='Sigmoid calibration (%1.3f)' % clf_sigmoid_score)
plt.ylim([-0.05, 1.05])
plt.xlabel("Instances sorted according to predicted probability ")
plt.ylabel("P(y=1)")
plt.legend(loc="lower right")
plt.title("Naive Bayes probabilities")        
remove_border()        
plt.show()

