"""
Classical SVM baseline.
"""

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


class ClassicalSVM:

    def __init__(self):

        self.model = SVC(kernel="rbf")

    def fit(self, X, y):

        self.model.fit(X, y)

    def predict(self, X):

        return self.model.predict(X)

    def accuracy(self, X, y):

        predictions = self.predict(X)

        return accuracy_score(y, predictions)