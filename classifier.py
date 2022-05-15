from abc import ABC, abstractmethod


class Classifier(ABC):
    @abstractmethod
    def train(self, x, y):
        """
        Learn the model from the training data.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predict the class of the test data.
        """
        pass

    @abstractmethod
    def get_accuracy(self, X, y_truth):
        """
        Using 5-fold cross validation, evaluate the model.
        """
        pass

    @abstractmethod
    def get_fairness(self, dataset):
        """
        Get fairness of the model.
        """
        pass

    @abstractmethod
    def model_status(self):
        """
        Return the model status.
        """
        pass
