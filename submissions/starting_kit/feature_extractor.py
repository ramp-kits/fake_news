# -*- coding: utf-8 -*-

from keras.preprocessing import sequence
from keras.preprocessing import text

class FeatureExtractor():
    
    tokenizer = text.Tokenizer()
    max_len = None
    
    def __init__(self):
        pass

    def fit(self, X_df, y):

        self.tokenizer.fit_on_texts(X_df.statement)        
        XX = self.tokenizer.texts_to_sequences(X_df.statement)        
        self.max_len = len(max(XX, key=len))
                
    def transform(self, X_df):
                
        XX = self.tokenizer.texts_to_sequences(X_df.statement)
        return sequence.pad_sequences(XX, maxlen=self.max_len)
