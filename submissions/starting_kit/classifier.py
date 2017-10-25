# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator
from sklearn.ensemble import (RandomForestClassifier,
                              VotingClassifier)
from sklearn.linear_model import LogisticRegression


class Classifier(BaseEstimator):
    def __init__(self):
        self.clf = VotingClassifier(estimators=[
            ('lr', LogisticRegression(solver='liblinear', max_iter=200)),
            ('rf', RandomForestClassifier(random_state=1, n_estimators=500))], voting='soft')

    def fit(self, X, y):
        self.clf.fit(X.todense(), y)

    def predict(self, X):
        return self.clf.predict(X.todense())

    def predict_proba(self, X):
        return self.clf.predict_proba(X)
