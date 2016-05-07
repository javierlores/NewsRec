import numpy as np


class LinearClassifier():
    """ 
        A base class for a linear classifier
    """

    def __init__(self):
        self.trained = False


    def train(self, X, y):
        """ 
            Trains the model

            Parameters
            ----------
            X: numpy array
                The training data
            y: numpy array
                Corresponding labels for the training data
        """
        raise NotImplementedError('LinearClassifier.train not implemented')


    def get_scores(self, X, w):
        """ 
            Computes the dot product between X and w

            Parameters
            ----------
            X: numpy array
                Samples to compute score
            w: numpy matrix
                Learned weight matrix
        """
        return np.dot(X, w)


    def get_label(self, X, w):
        """ 
            Computes the label for each data point

            Parameters
            ----------
            X: numpy array
                Samples to compute score
            w: numpy matrix
                Learned weight matrix
        """
        scores = np.dot(X, w)
        return np.argmax(scores, axis=1).transpose()


    def test(self, X, w):
        """ 
            Classifies the points based on a weight vector.

            Parameters
            ----------
            X: numpy array
                Samples to compute score
            w: numpy matrix
                Learned weight matrix
        """
        if self.trained == False:
            raise ValueError("Model not trained. Cannot test")
            return 0
        X = self.add_intercept_term(X)
        return self.get_label(X, w)

    
    def add_intercept_term(self, X):
        """ 
            Adds a column of ones to estimate the intercept term for separation boundary.

            Parameters
            ----------
            X: numpy array
                Samples to compute score
        """
        nr_x, nr_f = X.shape
        intercept = np.ones([nr_x, 1])
        X = np.hstack((intercept, X))
        return X


    def evaluate(self, truth, predicted):
        """ 
            Calculate the accuracy of the classifier.

            Parameters
            ----------
            truth: numpy array
                Correct labels for data
            predicted:
                Predicted labesl for data
        """
        correct = 0.0
        total = 0.0
        for i in range(len(truth)):
            if(truth[i] == predicted[i]):
                correct += 1
            total += 1
        return 1.0*correct/total


    def fscore(self, truth, predicted):
        """ 
            Calculate the fscore of the classifier.

            Parameters
            ----------
            truth: numpy array
                Correct labels for data
            predicted:
                Predicted labesl for data
        """
        # Build confusion matrix
        n_classes = len(np.unique(truth))
        confusion_matrix = [[0 for j in range(n_classes)] for i in range(n_classes)]
        for i in range(len(truth)):
            for j in range(n_classes):
                if predicted[i] == j:
                    for k in range(n_classes):
                       if truth[i] == k:
                           confusion_matrix[j][k] += 1

        # Calculate the F1-score of each class
        f1 = [0 for i in range(n_classes)]
        for i in range(n_classes):
            tp = confusion_matrix[i][i]                                  # True positives
            fn = sum([confusion_matrix[j][i] for j in range(n_classes)]) # False negatives
            fp = sum([confusion_matrix[i][j] for j in range(n_classes)]) # False positives

            recall = 1.0*tp/fn
            precision = 1.0*tp/fp

            f1[i] = 2.0*precision*recall / (precision+recall)

        return 1.0/n_classes*sum(f1)

            

            
        
