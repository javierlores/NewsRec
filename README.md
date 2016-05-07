NewsRec
===

This app is a class project that classifies news articles using multinomial naive bayes


Installation
---

In order to run this program, python 2.7 must be installed.


Dependencies
- numpy
- nltk

Running
---
#### Setup
You can download the dataset from http://qwone.com/~jason/20Newsgroups/ 
and choose the file named “20news-bydate.tar.gz"

Before running this program, the file 'topic_reader.py' needs to be edited
and the variables 'TRAIN_DATA_PATH' and 'TEST_DATA_PATH' need to be change
to the proper location of the training and testing files.

#### Execution
python run_topic_classifier.py

