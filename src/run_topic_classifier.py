
import time

from topic_reader import TopicCorpus
from multinomial_naive_bayes import MultinomialNaiveBayes


if __name__ == '__main__':
    start_time = time.time()

    # Create our dataset and model
    dataset = TopicCorpus()
    nb = MultinomialNaiveBayes()

    # Train the model
    params = nb.train(dataset.train_X, dataset.train_y)

    # Test the model on the training set
    predict_train = nb.test(dataset.train_X, params)
    eval_train = nb.evaluate(predict_train, dataset.train_y)
    fscore_train = nb.fscore(predict_train, dataset.train_y)
    
    # Test the model on the test set
    predict_test = nb.test(dataset.test_X, params)
    eval_test = nb.evaluate(predict_test, dataset.test_y)
    fscore_test = nb.fscore(predict_test, dataset.test_y)
    
    # Print results
    print "F1-Score on training set: %f, on test set: %f" % (fscore_train, fscore_test)
    print "Runtime: %s" % (time.time()-start_time)
