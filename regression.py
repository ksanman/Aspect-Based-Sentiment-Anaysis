from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

class Regression:
    """ Regression model. """
    def __init__(self):
        pass

    def fit(self, X, Y, x_test, y_test):

        cs = [0.01, 0.05, 0.25, 0.5, 1]
        accuracies = []
        for c in cs:
            
            lr = LogisticRegression(C=c, solver='sag')
            lr.fit(X, Y)
            accuracies.append(accuracy_score(y_test, lr.predict(x_test)))

        highestIndex = np.argmax(accuracies)
        print ("Regression Accuracy for C=%s: %s" % (cs[highestIndex], accuracies[highestIndex]))
	self.classifier = LogisticRegression(C=cs[highestIndex], solver='sag')
	self.classifier.fit(X,Y)

    def score(self, X, Y):
        return accuracy_score(Y, self.classifier.predict(X))

