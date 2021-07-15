from nltk import word_tokenize
from nltk.stem import PorterStemmer


class Stemmer(object):

    def __init__(self):
        self.wnl = PorterStemmer()

    def __call__(self, articles):
        return [self.wnl.stem(t) for t in word_tokenize(articles)]
