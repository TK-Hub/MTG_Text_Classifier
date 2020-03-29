#==================================================================================================
#                       Color Classification MTG - Application of Model
#                       Author: Tim
#                       Date: 14.03.2020
#==================================================================================================
from sklearn.ensemble import RandomForestClassifier
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

filename='mtg_model.sav'

def main(model_file, inp_sentence):
    sentence = Converter(inp_sentence)
    classifier = pickle.load(open(model_file, 'rb'))
    result = classifier.predict_proba(sentence)
    print(result)


def Converter(sentence):
    #tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
    tfidfconverter = pickle.load(open("tfidf.pickle", "rb"))
    Xt = tfidfconverter.transform(sentence).toarray()
    #print(Xt[0].shape)
    return Xt


if __name__ == "__main__":
    input_text=['Flying. Draw a card.', 'Fear. Destroy a creature.', 'You gain three life.', 'Add a colorless mana to your manapool.', 'Whenever an artifact enters the battlefield, scry 1.']
    #input_text = [input('Please enter text to classify:')]
    print(input_text)
    main(filename, input_text)