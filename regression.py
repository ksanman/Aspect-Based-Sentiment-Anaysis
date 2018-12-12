from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

class Regression:
    def __init__(self):
        pass

    def fit(self, X, Y):

        X_train, X_val, y_train, y_val = train_test_split(X, Y, train_size = 0.75)

        accuracies = []
        cs = [0.01, 0.05, 0.25, 0.5, 1]

        for c in cs:
            
            lr = LogisticRegression(C=c)
            lr.fit(X_train, y_train)
            accuracy = accuracy_score(y_val, lr.predict(X_val))
            print ("Accuracy for C=%s: %s" % (c, accuracy))
            accuracies.append(accuracy)

        self.C = cs[np.argmax(accuracies)]
        self.model = LogisticRegression(C = self.C)
        self.model.fit(X,Y)

    def evaluate(self, x):
        return self.model.predict(x)