# -*- coding: utf-8 -*-
#
#  Feature extractor
#  =================
#
#  The ``FeatureExtractor`` class us used to extract features
#  from text documents. It returns a sparse TF-IDF matrix.
#
#  The main approaches to customize the feature extraction
#  are,
#
#  1. Changing the initialization parameters of TfidfVectorizer
#     in the ``__init__`` function (see scikit-learn documentation).
#     This includes for instance, word n-grams, character n-grams,
#     other IDF weighting schemes etc.
#  2. Changing the document preprocessing in the
#     ``document_preprocessor`` function. This includes e.g. unicode
#     normalization.
#  3. Changing the token processing in the ``token_processor``
#     function. This includes e.g. stemming, lemmatization etc.
#     See NLTK package.


from __future__ import unicode_literals

from sklearn.feature_extraction.text import TfidfVectorizer


def document_preprocessor(doc):
    """ A custom document preprocessor

    This function can be edited to add some additional
    transformation on the documents prior to tokenization.

    At present, this function passes the document through
    without modification.
    """
    return doc


def token_processor(tokens):
    """ A custom token processor

    This function can be edited to add some additional
    transformation on the extracted tokens (e.g. stemming)

    At present, this function just passes the tokens through.
    """
    for t in tokens:
        yield t


class FeatureExtractor(TfidfVectorizer):
    """Convert a collection of raw docs to a matrix of TF-IDF features. """

    def __init__(self):
        # see ``TfidfVectorizer`` documentation for other feature
        # extraction parameters.
        super(FeatureExtractor, self).__init__(analyzer='word')

    def fit(self, X_df, y=None):
        """Learn a vocabulary dictionary of all tokens in the raw documents.

        Parameters
        ----------
        X_df : pandas.DataFrame
            a DataFrame, where the text data is stored in the ``statement``
            column.
        """
        super(FeatureExtractor, self).fit(X_df.statement)
        return self

    def fit_transform(self, X_df, y=None):
        return self.fit(X_df).transform(X_df)

    def transform(self, X_df):
        X = super(FeatureExtractor, self).transform(X_df.statement)
        return X

    def build_tokenizer(self):
        """
        Internal function, needed to plug-in the token processor, cf.
        http://scikit-learn.org/stable/modules/feature_extraction.html#customizing-the-vectorizer-classes
        """
        tokenize = super(FeatureExtractor, self).build_tokenizer()
        return lambda doc: list(token_processor(tokenize(doc)))