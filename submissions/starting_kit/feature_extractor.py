# -*- coding: utf-8 -*-

#import spacy
#import string as str_punctuation
#from nltk.corpus import stopwords
from keras.preprocessing import sequence
from keras.preprocessing import text
from sklearn import preprocessing

#english_stopwords = set(stopwords.words('english'))
#
#nlp = spacy.load('en')
#punctuation = set(str_punctuation.punctuation)

class FeatureExtractor():
    
    tokenizer = text.Tokenizer()
    max_len = None
#    label_encoder = preprocessing.LabelEncoder()
    
    def __init__(self):
        pass

    def fit(self, X_df, y):
        self.tokenizer.fit_on_texts(X_df.statement)
        
        XX = self.tokenizer.texts_to_sequences(X_df.statement)
        
        self.max_len = len(max(XX, key=len))
        
#        self.label_encoder.fit(y)
        
        


#    def tokenize(self, text):
#        doc = nlp(text)
#        return list((filter(lambda x: 
#                            x.text.lower() not in english_stopwords and
#                            not x.like_url
#                            and x.text.lower() not in punctuation,
#                            [t for t in [y for y in doc] if t.text.isalpha()])))
        
    def transform(self, X_df):
                
        XX = self.tokenizer.texts_to_sequences(X_df.statement)

        return sequence.pad_sequences(XX, maxlen=self.max_len)
