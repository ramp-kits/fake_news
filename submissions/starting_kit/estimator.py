from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline


def get_estimator():

    preprocessor = make_column_transformer(
        (TfidfVectorizer(analyzer='word'), 'statement'),
        remainder='drop',  # drop all other columns
    )
    clf = RandomForestClassifier()
    pipeline = make_pipeline(preprocessor, clf)
    return pipeline
