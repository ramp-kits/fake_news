import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit

problem_title = 'Fake News Detection'
_target_column_name = 'label'
_ignore_column_names = ['ID', 'true_counts', 'false_counts', 'half_true_counts', 
                        'mostly_true_counts', 'pants_on_fire']
_prediction_label_names = ['barely-true', 'false', 'half-true', 'mostly-true', 'pants-fire', 'true']
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)

# An object implementing the workflow
workflow = rw.workflows.FeatureExtractorClassifier()

score_types = [
    rw.score_types.Accuracy(name='acc'),
]


def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=8, test_size=0.2, random_state=57)
    return cv.split(X, y)

def _read_data(path, f_name):

    data = pd.read_csv(os.path.join(path, 'data', f_name), sep='\t', 
                       names=['ID', 'label', 'statement', 
                       'subject', 'speaker', 'job', 'state', 'party', 'true_counts', 
                       'false_counts', 'half_true_counts', 'mostly_true_counts', 
                       'pants_on_fire', 'context'])
    
    y_array = data['label'].values
    X_df = data.drop([_target_column_name] + _ignore_column_names, axis=1)
    
    return X_df, y_array


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)
#
#        X_train = feature_extractor.transform(X.ix[train_is])
#        X_test = feature_extractor.transform(X.ix[test_is])
#        
#        print("Training ..")
#        
#        clf = classifier.Classifier()
#        clf.fit(X_train, y_train)
#        
#        print('Testing..')
#        
#        y_predicted = clf.predict_proba(X_test)
#        
#        print(y_predicted)
#        print(y_test)
#        
#        auc = score_function(y_test, y_predicted[:,1])
#        print('auc = {}'.format(auc))
        