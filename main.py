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

REVIEW_COUNT = 1000

d_access = da.DataAccess()

training_set = d_access.get_training_reviews(REVIEW_COUNT)
testing_set = d_access.get_test_reviews(REVIEW_COUNT)

print 'Negative training data: ', len(training_set['negative'])
print 'Positive training data: ', len(training_set['positive'])
print 'Negative testing data: ', len(testing_set['negative'])
print 'Positive testing data: ', len(testing_set['positive'])

processor =  wp.WordProcessor()

tokenizer_training_reviews = training_set['positive'] + training_set['negative'] + testing_set['positive'] + testing_set['negative']

#positive_training_reviews = processor.train_tokenizer(tokenizer_training_reviews)
positive_training_reviews = processor.train_tokenizer()
