import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.util import pr


#Initializing Stemmer and Stopwords

stemmer=nltk.SnowballStemmer("english") #Stemmer is used to reduce words to their base form, and stopwords are words that are commonly used in a language and are not considered significant for analysis.
from nltk.corpus import stopwords
import string
stopword=set(stopwords.words("english"))


#Loading the Data

data=pd.read_csv("twitter_data.csv")


#Preprocessing the Data

data['labels']=data['class'].map({0:"Hate speech",1:"Not offensive",2:"Neutral"})
data = data[["tweet","labels"]] #we select only the tweet and labels columns for further processing.


#Tokenization and Stemming

def tokenize_stem(text):  #"tokenize_stem" function tokenizes the text (splits it into words), removes stopwords and non-alphabetic characters, and then stems the words using the stemmer.
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stopword]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [stemmer.stem(word) for word in tokens]
    return tokens


#Vectorization

vectorizer = CountVectorizer(tokenizer=tokenize_stem) #creates a CountVectorizer object with the tokenize_stem function as the tokenizer.
X = vectorizer.fit_transform(data['tweet']) #fits and transforms the tweet column of the data DataFrame into a sparse matrix X.


#Splitting the Data

X_train, X_test, y_train, y_test = train_test_split(X, data['labels'], test_size=0.2, random_state=42)
#splits the data into training and testing sets. X_train and y_train contain the features and labels for the training set, while X_test and y_test contain the features and labels for the testing set.


#Training the Classifier\

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)


#Taking User Input

tweet = input("Enter your tweet: ")


#Vectorizing User Input:

tweet_vector = vectorizer.transform([tweet]) #vectorizes the user input tweet using the CountVectorizer object.


#Predicting the Label

prediction = classifier.predict(tweet_vector)
print(f"The tweet is classified as: {prediction[0]}")

