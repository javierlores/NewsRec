import numpy as np


from linear_classifier import LinearClassifier


class MultinomialNaiveBayes(LinearClassifier):
    """ 
        A Multinomial Naive Bayes model.
    """
    def __init__(self):
        LinearClassifier.__init__(self)
        self.trained = False
        self.likelihood = 0
        self.prior = 0
        self.smooth = True
        self.smooth_param = 1
        

    def train(self, x, y):
        """ 
            This function trains the model.

            Parameters
            ----------
            x: numpy array
                The training data
            y: numpy array
                The corresponding labels
        """
        # n_docs = no. of documents
        # n_words = no. of unique words    
        n_docs, n_words = x.shape
        
        # classes = a list of possible classes
        classes = np.unique(y)
        
        # n_classes = no. of classes
        n_classes = np.unique(y).shape[0]
        
        # initialization of the prior and likelihood variables
        prior = np.zeros(n_classes)
        likelihood = np.zeros((n_words,n_classes))

        # Generate prior probabilities
        for i in range(n_classes):
            prior[i] = 1.0 * len(np.where(y==i)[1]) / n_docs

        # Generate likelihood probabilities
        if self.smooth:
            denominator = [np.sum(np.where(y==j)[0])+(n_words*self.smooth_param) for j in range(n_classes)]
            numerator = [np.sum(x[np.where(y==j)[0]], axis=0)+self.smooth_param for j in range(n_classes)]
        else:
            denominator = [np.sum(np.where(y==j)[0]) for j in range(n_classes)]
            numerator = [np.sum(x[np.where(y==j)[0]], axis=0) for j in range(n_classes)]

        for j in range(n_classes):
            likelihood[..., j] = 1.0 * numerator[j] / denominator[j]

        # Compute Log probabilities
        params = np.zeros((n_words+1, n_classes))
        for i in range(n_classes): 
            params[0,i] = np.log(prior[i])
            with np.errstate(divide='ignore'): # ignore warnings
                params[1:,i] = np.nan_to_num(np.log(likelihood[:,i]))
        self.likelihood = likelihood
        self.prior = prior
        self.trained = True
        return params
