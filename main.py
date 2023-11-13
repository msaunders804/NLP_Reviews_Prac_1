#Basics
import numpy as np
import pandas as pd
#Text clean up
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer as PS
#Stop words are words that do not typically add to intent ex. the, a ,an,in
nltk.download("stopwords")
#Tokenization
from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.model_selection import train_test_split as tts
#fit
from sklearn.ensemble import RandomForestClassifier as RFC
#Accuracy
from sklearn.metrics import confusion_matrix as CM
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t')

#clean up text
clean=[]

for i in range(0, 1000):
    #Pull out the review column from data
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    #Convert to all lowercase
    review = review.lower()
    #Split into an array (split on " " by default)
    review = review.split()
    #create a Porter Stemmer object
    #A porter stemmer is a normalizer that removes the infexional endings from words ie batting to bat
    ps = PS()
    #stem all the words in review, except stop words those get remove
    review = [ps.stem(word) for word in review
              if not word in set(stopwords.words('english'))]

    #print(review) -- at this point we have and list of arrays that contain condensed text reivews
    #Bring them back together to be strings
    review = ' '.join(review)
    clean.append(review)

#Tokenization
cv = CV(max_features=1500)
#Fit and transform our cleaned up reviews; creates a bag of words
X = cv.fit_transform(clean).toarray()
#Pull our column values at index 1; Which is the 0 or 1 of negative or postive
y = dataset.iloc[:,1].values

#Splitting with 30% being the test
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3)

#Fit
model = RFC(n_estimators=500, criterion='entropy')
model.fit(X_train,y_train)

#predict -- following this we will have y_pred
#array that lists 0 or 1 based on the predicition the review is postive or negative
y_pred = model.predict(X_test)

#Now we need to see how accurate it is
cf = CM(y_test, y_pred)
#This results in a confusion matrix that displays the True pos, true neg, false neg and true neg

print(accuracy_score(y_test,y_pred))