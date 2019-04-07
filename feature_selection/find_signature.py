#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)
import sys
sys.path.append("../tools/")
sys.path.append("../choose_your_own")
sys.path.append("../datasets_questions")

import os
os.chdir('../feature_selection')

### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
#words_file = "../text_learning/your_word_data.pkl" 
#authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open("../text_learning/your_word_data.pkl", "rb"))
authors = pickle.load( open("../text_learning/your_email_authors.pkl", "rb") )

def my_func(words_file, authors_file):
    '''
    I will use this code later in the lesson so I made it a function
    '''
    
    ### The words (features) and authors (labels), already largely processed.
    ### These files should have been created from the previous (Lesson 10)
    ### mini-project.
    word_data = pickle.load( open(words_file, "rb"))
    authors = pickle.load( open(authors_file, "rb") )


### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
    from sklearn import cross_validation
    features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
    features_train = vectorizer.fit_transform(features_train)
    features_test  = vectorizer.transform(features_test)


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
    features_train = features_train[:150].toarray()
    labels_train   = labels_train[:150]



### your code goes here
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier()
    clf.fit(features_train, labels_train)

    return clf, vectorizer, features_train, features_test, labels_train, labels_test

(clf, vectorizer, features_train, features_test, labels_train, labels_test) = my_func("../text_learning/your_word_data.pkl", "../text_learning/your_email_authors.pkl")

print('Number of training points = {0}'.format(len(features_train)))

print('Accuracy on test set = {0}'.format(clf.score(features_test, labels_test)))

# What is the importance of the most important feature? What is the number of this feature?
import numpy as np
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
print('Feature Ranking: ')
for i in range(10):
    print("{} feature no.{} ({})".format(i+1, indices[i], importances[indices[i]]))  # feature no.33614 (0.764705882353)

top_features = [(number, feature, vectorizer.get_feature_names()[number]) for number, feature in 
                zip(range(len(clf.feature_importances_)), clf.feature_importances_) if feature > 0.2]
print(top_features)    

# What is the most powerful word when your decision tree is making its classification decisions?
print(vectorizer.get_feature_names()[33614])  # 'sshacklensf'

#Remove, Repeat
import string
from nltk.stem.snowball import SnowballStemmer

def parseOutText(f):
    '''
    Input: a file containing text
    
    Output: the stemmed words in the input text, all separated by a single space
    '''
    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")
    
    # the stemmer
    stemmer = SnowballStemmer('english')
    
    # the string of words
    words = ""
    
    if len(content) > 1:
        ### remove punctuation
        text_string = content[1].translate(string.maketrans("", ""), string.punctuation)

        ### split the text string into individual words, stem each word,
        ### and append the stemmed word to words (make sure there's a single
        ### space between each stemmed word)
        for word in text_string.split():
            # stem the word and add it to words
            words += stemmer.stem(word) + ' '       
        
    return words[:-1]
    

ff = open("../text_learning/test_email.txt", "r")
text = parseOutText(ff)

#from sklearn.feature_extraction.text import TfidfVectorizer

"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""

def sara_and_chris(sw):
    '''
    I'm going to reuse this code later so I'm making it a function
    '''
    
    with open("../text_learning/from_sara.txt", "r") as from_sara, open("../text_learning/from_chris.txt", "r") as from_chris:

        from_data = []
        word_data = []

        ### temp_counter is a way to speed up the development--there are
        ### thousands of emails from Sara and Chris, so running over all of them
        ### can take a long time
        ### temp_counter helps you only look at the first 200 emails in the list so you
        ### can iterate your modifications quicker
        temp_counter = 0


        for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
            for path in from_person:
                ### only look at first 200 emails when developing
                ### once everything is working, remove this line to run over full dataset

                #temp_counter += 1
                if temp_counter < 200:
                    path = os.path.join('..', path[:-1])

                    with open(path, 'r') as email:
                        ### use parseOutText to extract the text from the opened email
                        text = parseOutText(email)

                        ### use str.replace() to remove any instances of the words
                        ### ["sara", "shackleton ", "chris", "germani"]
                        for word in sw:
                            if(word in text):
                                text = text.replace(word, "")

                        ### append the text to word_data
                        word_data.append(text.replace('\n',' ').strip())

                        ### append a 0 to from_data if email is from Sara, and 1 if email is from Chris
                        if name=='sara':
                            from_data.append(0)
                        else:
                            from_data.append(1)

    pickle.dump( word_data, open("your_word_data.pkl", "w") )
    pickle.dump( from_data, open("your_email_authors.pkl", "w") )
    
    return None


sw = ["sara", "shackleton", "chris", "germani", "sshacklensf"]
sara_and_chris(sw)    



#Re-Fit the Model and Find the Outlier
words_file = 'your_word_data.pkl'
authors_file = 'your_email_authors.pkl'

(clf, vectorizer, features_train, features_test, labels_train, labels_test) = my_func(words_file, authors_file)

top_features = [(number, feature, vectorizer.get_feature_names()[number]) for number, feature in 
                zip(range(len(clf.feature_importances_)), clf.feature_importances_) if feature > 0.2]
print(top_features)

#Accuracy of the Overfit Tree
# remove the 2 outlier words
sw = ["sara", "shackleton", "chris", "germani", "sshacklensf", "cgermannsf"]
sara_and_chris(sw)

# re-fit the tree
words_file = 'your_word_data.pkl'
authors_file = 'your_email_authors.pkl'
(clf, vectorizer, features_train, features_test, labels_train, labels_test) = my_func(words_file, authors_file)

print('Accuracy on test set = {0}'.format(clf.score(features_test, labels_test)))