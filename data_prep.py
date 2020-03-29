#==================================================================================================
#                       Color Classification MTG - Data Preparation
#                       Author: Tim
#                       Date: 14.03.2020
#                       Reference: https://stackabuse.com/text-classification-with-python-and-scikit-learn/
#==================================================================================================

import json
import pandas as pd
import nltk
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.stem import WordNetLemmatizer

#Initial Definitions
minidict=[]

def json_to_dframe():
    with open("LegacyCards.json", encoding="utf8") as json_file:
        data = json.load(json_file)
        for key, value in data.items():
            if 'text' in value:
                if len(value['colorIdentity'])==1:
                    minidict.append([key, value['text'], value['colorIdentity'][0]])
                elif len(value['colorIdentity'])==0:
                    minidict.append([key, value['text'], 'L'])
                else:
                    #minidict.append([key, value['text'], value['colorIdentity'][0]])
                    pass

    dframe = pd.DataFrame(minidict, columns=['Card', 'Text', 'Color'])
    print(dframe)
    return dframe

def text_conversion(dataframe):
    X, y = dataframe['Text'].tolist(), dataframe['Color'].tolist()
    stemmer = WordNetLemmatizer()
    #nltk.download('wordnet')

    for text in X:
        # Converting text to lowercase
        text = text.lower()
        # Lemmatization
        text = text.split()
        text = [stemmer.lemmatize(word) for word in text]
        text = ' '.join(text)
    #print(X)
    return X, y


def train_model(X, y):
    # These are two different approaches, tfidf juest goes a step further and takes the appearance number of a word into account. I used the direct approach below.
    '''vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
    vectors = vectorizer.fit_transform(X).toarray()
    
    tfidfconverter = TfidfTransformer()
    X = tfidfconverter.fit_transform(vectors).toarray()'''
    
    # Initialize Vectorizer
    tfidfconverter = TfidfVectorizer(max_features=1500, min_df=3, max_df=0.7, stop_words=stopwords.words('english'))
    
    # Train the vectorizer and save it for the re-application
    tfidf = tfidfconverter.fit(X)
    pickle.dump(tfidf, open("tfidf.pickle", "wb"))

    # Convert the present data set into vectors
    Xt = tfidfconverter.transform(X).toarray()
    X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=0.2, random_state=0)

    classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
    classifier.fit(X_train, y_train)

    print(X_test)
    y_pred = classifier.predict(X_test)
    
    print(X_test[0].shape)
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print(accuracy_score(y_test, y_pred))

    # save the model to disk
    filename = 'mtg_model.sav'
    pickle.dump(classifier, open(filename, 'wb'))


#==================================================================================================
if __name__ == "__main__":
    texts = json_to_dframe()
    X, y = text_conversion(texts)
    train_model(X,y)

