from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class RandomForest:
    """ Random forest model. """
    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=0)

    def fit(self, x_train, y_train, x_test, y_test):
        self.classifier.fit(x_train, y_train)
        accuracy = accuracy_score(y_test, self.classifier.predict(x_test))
	print 'Random forest accuracy: ', accuracy

    def score(self, X,Y):
        return  accuracy_score(Y, self.classifier.predict(X))

