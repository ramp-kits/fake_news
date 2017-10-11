# -*- coding: utf-8 -*-

import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Model
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, Input
from keras.utils import np_utils
from keras.layers.merge import concatenate

from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

import numpy as np

import matplotlib
matplotlib.rcParams['backend'] = "Agg"
#matplotlib.use('TkAgg')
import keras
from sklearn.base import BaseEstimator
from sklearn import preprocessing

class Classifier(BaseEstimator):
    
    label_encoder = preprocessing.LabelEncoder()
    
    N_OUT = None
    MAX_SEQUENCE_LENGTH = None
    N_WORDS = None
    EMBEDDING_SIZE = 300
    
    def __init__(self):
        return None
    
    def initialize(self):
        
        embedding_layer = Embedding(self.N_WORDS,
            self.EMBEDDING_SIZE,
            embeddings_initializer=keras.initializers.glorot_normal(seed=189),
            input_length=self.MAX_SEQUENCE_LENGTH,
            trainable=True, name='embedded_words')
    
        ngram_filters = [2, 3, 4, 5]
        filter_size = 300
        graph_in = Input(shape=(self.MAX_SEQUENCE_LENGTH, self.EMBEDDING_SIZE))
        convs = []
        for fsz in ngram_filters:
            conv = Conv1D(filters=filter_size,
                          kernel_size=fsz,
                          padding='valid',
                          activation='relu',
                          strides=1)(graph_in)
            pool = GlobalMaxPooling1D()(conv)
            convs.append(pool)
        if len(ngram_filters) > 1:
            out = concatenate(convs)
        else:
            out = convs[0]
        
        convolutions = Model(inputs=graph_in, outputs=out, name='convolutions')
        
        sequence_sent_input = Input(shape=(self.MAX_SEQUENCE_LENGTH,), dtype='int32', name='sequence_words')
        embedded_sent = embedding_layer(sequence_sent_input)
        print(embedded_sent.shape)
        convs = convolutions(embedded_sent)
        drop = Dropout(0.5, name='dropout_0.5')(convs)
        output = Dense(self.N_OUT, activation='softmax', name='softmax')(drop)
        model = Model(inputs=[sequence_sent_input], \
            outputs=[output])
            
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        self.clf = model


    def fit(self, X, y):

        self.MAX_SEQUENCE_LENGTH = X.shape[1]
        self.N_WORDS = 30000
        y = self.label_encoder.fit_transform(y)
        self.N_OUT = max(y)+1
        y = np_utils.to_categorical(y, max(y)+1)
        
        self.initialize()
        
        self.clf.fit(X, y,
                  validation_split=0.7, 
                  shuffle=True, 
                  batch_size=150, 
                  verbose=1, 
                  epochs=2)

    def predict_proba(self, X):
        return self.clf.predict(X)
