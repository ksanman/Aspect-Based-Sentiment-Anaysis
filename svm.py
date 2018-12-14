from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

class SVM:
    def __init__(self):
        self.classifier = SVC(kernel='linear')

    def fit(self, x_train, y_train, x_test, y_test):
	self.classifier.fit(x_train, y_train)
	accuracy = accuracy_score(y_test, self.classifier.predict(x_test))
	print 'SVM Accuracy: ', accuracy


    def score(self, X,Y):
    	return accuracy_score(Y, self.classifier.predict(X))

