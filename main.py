"""
The main file to run the program. 
Author: Kody Sanchez
Course: CS5600-001 Intelligent Systems
Project 2
Instructor: Vladimr Kulyukin
            
Aspect Based Sentiment Analysis Classifier Comparision. 
"""

import word_processor as wp
import data_access as da
import numpy as np
import regression as r


REVIEW_COUNT = 1000

d_access = da.DataAccess()

training_set = d_access.get_training_reviews(REVIEW_COUNT)
testing_set = d_access.get_test_reviews(REVIEW_COUNT)

print 'Negative training data: ', len(training_set['negative'])
print 'Positive training data: ', len(training_set['positive'])
print 'Negative testing data: ', len(testing_set['negative'])
print 'Positive testing data: ', len(testing_set['positive'])

processor =  wp.WordProcessor()

train_pos_labels = np.ones(len(training_set['positive']), dtype=int)
train_neg_labels = np.zeros(len(training_set['negative']),dtype=int)
test_pos_labels = np.zeros(len(testing_set['positive']),dtype=int)
test_neg_labels = np.ones(len(testing_set['negative']),dtype=int)

training_labels = np.concatenate((train_pos_labels, train_neg_labels), axis=0)
training_data = np.concatenate((training_set['positive'], training_set['negative']), axis=0)

testing_labels = np.concatenate((test_pos_labels,test_neg_labels), axis=0)
testing_data = np.concatenate((testing_set['positive'],testing_set['negative']), axis=0)

training_data_clean = processor.process(training_data)
testing_data_clean = processor.process(testing_data)

X = processor.vectorize_train(training_data_clean)
T = processor.vectorize(testing_data_clean)
reg = r.Regression()
reg.fit(X, training_labels)

results = []
for test_review in T:
    results.append(reg.evaluate(test_review))

acc = 0.0
for i, res in enumerate(results):
    if res[0] == testing_labels[i]:
        acc += 1.0

n = len(testing_labels)
f_acc = acc / n
print 'Final Regression Accuracy: ', f_acc * 100