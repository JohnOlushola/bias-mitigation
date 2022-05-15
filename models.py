from numpy import mean, std
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score

from classifier import Classifier
from dataset import Dataset
from fairness_metrics import Fairness


class LogisticRegressionClassifier(Classifier):
    def __init__(
        self,
        verbose=False,
        solver="liblinear",
        params={},
        dataset=None,
        privileged_groups=None,
        unprivileged_groups=None,
        shuffle=False,
    ):
        self.verbose = verbose
        self.shuffle = shuffle
        self.cv = KFold(n_splits=5, random_state=None, shuffle=shuffle)
        self.model = LogisticRegression(
            verbose=verbose, solver=solver, **params, random_state=1
        )
        self.privileged_groups = privileged_groups
        self.unprivileged_groups = unprivileged_groups
        self.dataset: Dataset = dataset

        # validating k
        self.dataset_splits = []  # dataset splits for fairness
        self.current_k = 0  # util variable for tracking what fold is being validated

        if verbose:
            print("Model initialized: \n" + str(self.model.get_params()))

    def train(self, x: any = None, y: any = None, instance_weights=None):
        """
        Train the model on the given data.

        :param x: The data to train on.
        :param y: The labels for the data.
        :param instance_weights: The weights for each instance.
        """
        self.model.fit(x, y, sample_weight=instance_weights)

    def predict(self, x=None):
        """
        Predict the labels for the given data.

        :param x: The data to predict on.
        """
        return self.model.predict(x)

    def draw(self, type="model", X=None, y=None, dataset=None):
        """
        Returns a plot of the model's decision boundary.
        """
        X_train, y_train, X_test, y_test = self.dataset.get_data()
        X = X_train if X is None else X
        y = y_train if y is None else y

        if type == "model":
            return "Not implemented"
        elif type == "confusion_matrix":
            pred = self.predict(X)
            cm = confusion_matrix(y, pred)

            disp = ConfusionMatrixDisplay(cm).from_estimator(self.model, X=X, y=y)

            plt.figure(figsize=(10, 10))
            return disp.plot()

        else:
            return print("Type not implemented")

    def model_status(self):
        print("Not implemented")

    def __fairness_scoring__(self, clf: Classifier = None, X=None, y=None):
        """
        Returns Score as True Positive Rate Difference
        """
        # current split
        split = self.dataset_splits[self.current_k]

        equal_opp_diff = self.get_fairness(X=split, metric="eq_opp_diff")

        # increment k
        self.current_k += 1

        return equal_opp_diff

    def __delta_scoring__(self, clf: Classifier = None, X=None, y=None):
        """
        This accounts for both accuracy and fairness
        """
        # current split
        split = self.dataset_splits[self.current_k]

        equal_opp_diff = self.get_fairness(X=split, metric="eq_opp_diff")

        # # pass equal_opp_diff to sigmoid function to scale to [0,1]
        # equal_opp_diff = 1 / (1 + math.exp(-equal_opp_diff))

        accuracy = self.get_accuracy(X=X, y_truth=y)

        # increment k
        self.current_k += 1

        return accuracy - equal_opp_diff

    def cross_validation(self, X, y, method="accuracy"):
        """
        Get the cross validation score for the model over 5 folds.
        """
        if method == "accuracy":
            scores = cross_val_score(self.model, X, y, cv=self.cv, scoring="accuracy")
            if self.verbose:
                print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
            return mean(scores), std(scores)

        elif method == "fairness":
            if self.privileged_groups is None or self.unprivileged_groups is None:
                return print("No Priviledge or unpriviledged groups provided")

            # split dataset into k folds, k = 5
            self.dataset_splits = self.dataset.dataset_orig_train.split(
                5, shuffle=self.shuffle
            )

            scores = cross_val_score(
                self.model, X, y, cv=self.cv, scoring=self.__fairness_scoring__
            )

            if self.verbose:
                print(str(scores))

            self.current_k = 0
            return mean(scores)
        elif method == "accuracy+fairness":
            if self.privileged_groups is None or self.unprivileged_groups is None:
                return print("No Priviledge or unpriviledged groups provided")

            # split dataset into k folds, k = 5
            self.dataset_splits = self.dataset.dataset_orig_train.split(
                5, shuffle=self.shuffle
            )

            scores = cross_val_score(
                self.model, X, y, cv=self.cv, scoring=self.__delta_scoring__
            )

            if self.verbose:
                print(str(scores))

            self.current_k = 0
            return mean(scores)
        else:
            return print("Method not implemented")

    def get_accuracy(self, X, y_truth):
        """
        Get accuracy of the model

        :param X: The data to evaluate on.
        :param y_truth: True label of X.
        """
        pred = self.predict(X)

        return mean(pred == y_truth)

    def get_fairness(self, X: Dataset = None, metric=None):
        """
        Get fairness metric value(s) of the model

        :param X: The data to evaluate on.
        :param metric: Exact fairness metric to return.

        :return: Fairness metric value(s)
        """
        if X is None:
            return print("No dataset provided")

        X_test, y_test = self.dataset.scale(X)

        pred = self.predict(X_test)
        y_pred = X.copy()
        y_pred.labels = pred

        fairness = Fairness(
            X=X,
            y=y_pred,
            unprivileged_groups=self.unprivileged_groups,
            privileged_groups=self.privileged_groups,
        )
        metrics = fairness.get_metrics()

        return metrics[metric] if metric else metrics

    def get_accuracy_fairness(self, X, y_truth):
        """
        Get accuracy+fairness(delta) value of the model

        delta = accuracy - fairness

        :param X: The data to evaluate on.
        :param y_truth: True label of X.

        :return: accuracy+fairness value
        """

        return self.get_accuracy(X, y_truth) - self.get_fairness(X)

    def run(
        self,
        dataset: Dataset = None,
        model=None,
        verbose=False,
        mitigated=False,
        include_delta=False,
    ): 
        """
        Trains the model and returns the accuracy, fairness and/or accuracy+fairness (delta) score.

        :param dataset: The dataset to train on.
        :param model: The model to train.
        :param verbose: Prints the accuracy, fairness and/or accuracy+fairness (delta) score.
        :param mitigated: If True, the model is trained on the mitigated dataset.
        :param include_delta: If True, the model is trained on the mitigated dataset.

        :return: Accuracy, fairness and/or accuracy+fairness (delta) score.
        """
        if dataset is None:
            return print("No dataset provided")

        if model is None:
            model = self

        # get data
        X_train, y_train, X_test, y_test = dataset.get_data()

        # train model
        if verbose:
            print("Training model...")
        model.train(
            X_train,
            y_train,
            instance_weights=dataset.dataset_orig_train.instance_weights
            if mitigated
            else None,
        )

        # get accuracy and fairness
        if verbose:
            print("Computing accuracy and fairness across 5-fold cross validation...")

        score, std = model.cross_validation(X_train, y_train, method="accuracy")
        fairness_metrics = model.cross_validation(X_train, y_train, method="fairness")

        # get delta
        if include_delta:
            accuracy_fairness_metrics = model.cross_validation(
                X_train, y_train, method="accuracy+fairness"
            )

        if verbose:
            print(
                "Training data: Accuracy -> %0.2f (+/- %0.2f)" % (score, std * 2),
                end="\n\n",
            )
            print("Training data: Fairness metric -> \n" + str(fairness_metrics))
            if include_delta:
                print(
                    "Training data: Accuracy+Fairness metric -> \n"
                    + str(accuracy_fairness_metrics)
                )

        values = (
            (score, std, fairness_metrics, accuracy_fairness_metrics)
            if include_delta
            else (score, std, fairness_metrics)
        )

        return values

    def get_params(self):
        """
        Get the parameters of the model.
        """
        return self.model.get_params()

