"""
The main file to run the program. 
Author: Kody Sanchez
Course: CS5600-001 Intelligent Systems
Project 2
Instructor: Vladimir Kulyukin
            
Aspect Based Sentiment Analysis Classifier Comparision. 
"""

import word_processor as wp
import data_access as da
import numpy as np
import regression as r
import random_forest as rf
import svm
import ann
import cnn
import rnn

from sklearn.model_selection import train_test_split

#Constants
SEED = 2000

#################################################
#
# These are the environment variables that dictate how the program runs. 
# You only have to change these variables. 
#
#################################################

# Lower this number when training the CNN and RNN, otherwise there will be an out of memory exception. 
# The RNN and CNN were tested with 100 reviews and 80 features. 
REVIEW_COUNT = 100000
FEATURES = 100000
DATA_DIRECTORY = '../review_data/'

###################################################


d_access = da.DataAccess()
data = d_access.get_data(REVIEW_COUNT,DATA_DIRECTORY)
processor =  wp.WordProcessor(FEATURES)

print 'positive data: ', len(data['positive'])
print 'negative data: ', len(data['negative'])
pos_labels = [1 for _ in range(len(data['positive']))]
neg_labels = [0  for _ in range(len(data['negative']))]


labels = np.concatenate([pos_labels, neg_labels])
inputs = np.concatenate([data['positive'], data['negative']])

# Preprocess the data
inputs = processor.process(inputs)

x_train, x_leftover, y_train, y_leftover = train_test_split(inputs, labels, test_size=0.25, train_size=0.75,random_state=SEED)

x_validation, x_test, y_validation, y_test = train_test_split(x_leftover, y_leftover, test_size=0.5, train_size=0.5, random_state=SEED)

x_vect = processor.vectorize_train(x_train)
x_val_arr = processor.vectorize(x_validation).toarray()
x_test_arr = processor.vectorize(x_test).toarray()
#print x_vect
#print x_vect.toarray()

#f = open('results1.txt', 'w+')

print 'linear regression'
reg = r.Regression()
reg.fit(x_vect, y_train, x_val_arr, y_validation)
score = reg.score(x_test_arr, y_test)
print 'Regression Score: ', score
#f.write('Regression Score: {0}\n'.format(score))

print 'random forest'
forest = rf.RandomForest()
forest.fit(x_vect, y_train, x_val_arr, y_validation)
score = forest.score(x_test_arr, y_test)
print 'Random Forest Score: ', score
#f.write('Random Forest Score: {0}\n'.format(score))

print 'svm'
s = svm.SVM()
s.fit(x_vect, y_train, x_val_arr, y_validation)
score = s.score(x_test_arr, y_test)
print 'svm accuracy: ',score 
#f.write('SVM Score: {0}\n'.format(score))

def one_hot(Y):
   return [[0, 1] if y == 1 else [1,0] for y in Y] 

print 'ann'

nn = ann.ANN()
nn.build_model(x_vect.toarray(), one_hot(y_train), x_val_arr, one_hot(y_validation))

score = nn.score(x_test_arr, y_test)
print 'ANN accuracy: ', score
#f.write('ANN Score: {0}\n'.format(score))

##############################################################################
#
# Seg-fault errors and out of memory exceptions started being thrown when 
# training at 100,000 reviews. Bringing the number down did not resolve these
# errors. THESE MODELS ARE NOT TESTED AND NOT GUARANTEED TO WORK. 
#
##############################################################################

# print 'cnn'

# c_nn = cnn.CNN()
# c_nn.build_model(x_vect.toarray(), one_hot(y_train), x_val_arr, one_hot(y_validation))

# score = c_nn.score(x_test_arr, y_test)
# print 'CNN accuracy: ', score
# f.write('CNN Score: {0}\n'.format(score))

# print 'rnn'

# r_nn = rnn.RNN()
# r_nn.build_model(x_vect.toarray(), one_hot(y_train), x_val_arr, one_hot(y_validation))

# score = r_nn.score(x_test_arr, y_test)
# print 'RNN accuracy: ', score
# f.write('RNN Score: {0}\n'.format(score))
# f.close()



