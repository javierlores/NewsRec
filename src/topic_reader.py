import codecs
import numpy as np
import random
import nltk.stem
import nltk.tokenize
import nltk.corpus
import re
import itertools
import collections
import os


TRAIN_DATA_PATH = "../data/20news-bydate-train/"
TEST_DATA_PATH = "../data/20news-bydate-test/"
TOPICS = ["comp.graphics", "rec.autos", "talk.politics.guns"]


class TopicCorpus:
    """ 
        Class to encapuslate the dataset
    """
    def __init__(self):
        """ 
            prepare dataset
            1) Read train and test sets and preprocess
            2) Create vocabulary
            3) Count word frequencies for train and test sets
        """
        self.train_X, self.train_y = self._read_data(TRAIN_DATA_PATH)
        self.test_X, self.test_y = self._read_data(TEST_DATA_PATH)

        self._create_features(self.train_X)

        self.train_X, self.train_y  = self._extract_features(self.train_X, self.train_y)
        self.test_X, self.test_y  = self._extract_features(self.test_X, self.test_y)


    def _read_data(self, path):
        """ 
            This function reads in the dataset

            Parameters
            ----------
            path: str
                The path to the dataset
        """
        data_X = [] # The training data
        data_y = [] # The training labels

        # Read in the data
        for index, topic in enumerate(TOPICS):
            for file in os.listdir(path+topic):
                data_X.append(self._preprocess(path+topic+'/'+file))
                data_y.append([index])

        return data_X, data_y
                 

    def _preprocess(self, file):
        """ 
            This function preprocesses a document

            Parameters
            ----------
            file: str
                the filename of the dataset
        """
        with codecs.open(file, 'r', encoding='utf8', errors='ignore') as doc:
            data = doc.read() # Document data

            # Sentence segmentation
            sentence_tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
            data = sentence_tokenizer.tokenize(data)

            # Word tokenization
            word_tokenizer = nltk.tokenize.WordPunctTokenizer()
            data = [word_tokenizer.tokenize(sentence) for sentence in data]

            # Remove words with less than 5 characters
            data = [[word for word in sentence if len(word) > 5] for sentence in data]

            # Remove words that only contain digits
            data = [[word for word in sentence if not word.isdigit()] for sentence in data]

            # Remove words that only contain punctuation
            data = [[word for word in sentence if not re.match(r'^[\W_]+$', word)] for sentence in data]

            # Word lowercasing
            data = [[word.lower() for word in sentence] for sentence in data]

            # Lemmatization
            word_lemmatizer = nltk.stem.WordNetLemmatizer()
            data = [[word_lemmatizer.lemmatize(word) for word in sentence] for sentence in data]

            # Stopword removal
            stopwords = set(nltk.corpus.stopwords.words('english'))
            data = [[word for word in sentence if word not in stopwords] for sentence in data]

        return data


    def _create_features(self, X):
        """ 
            This function creates the vocabulary of the BOW representation

            Parameters
            ----------
            X: numpy array
                The documents
        """
        # Create the vocabulary
        words = list(itertools.chain.from_iterable(itertools.chain.from_iterable(X)))
        self.vocabulary = set(np.unique(np.array(words)).tolist())

        # Create a template of the vocabulary that will hold the counts of each word in the doc
        counts = np.zeros(len(self.vocabulary)).astype('int32')
        self.feature_template = dict(zip(self.vocabulary, counts.tolist()))


    def _extract_features(self, X, y):
        """ 
            This function extracts extracts the BOW feature for each document

            Parameters
            ----------
            X: numpy array
                The documents
            y: numpy array
                The label for each document
        """
        # Extract the features for each document
        features = []
        for doc in X:
            feature = self.feature_template.copy()
            for sentence in doc:
                for word in sentence:
                    if word in self.vocabulary:
                        feature[word] += 1
            features.append(feature.values())
                            
        # shuffle the order
        features = np.array(features)
        y = np.array(y)
        new_order = np.arange(features.shape[0])
        np.random.seed(0) # set seed
        np.random.shuffle(new_order)
        features = features[new_order,:]
        y = y[new_order]

        return features, y
