from sklearn.preprocessing import StandardScaler
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.datasets import AdultDataset, Dataset, GermanDataset
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import (
    load_preproc_data_adult,
    load_preproc_data_german,
)
from aif360.algorithms.preprocessing import Reweighing
import numpy as np
import sys

sys.path.insert(1, "../")


np.random.seed(0)


# from aif360.algorithms.inprocessing import AdversarialDebiasing


class Dataset:
    def __init__(self, dataset_name="adult", protected_attributes_names=["sex"]):

        # init dataset
        if dataset_name == "adult":
            self.dataset_orig = load_preproc_data_adult(protected_attributes_names)
            print("Adult Dataset initialized")
        elif dataset_name == "german":
            self.dataset_orig = load_preproc_data_german(protected_attributes_names)
            print("German Dataset initialized")
        else:
            print(
                "Instance not functional: Dataset not implemented, please choose adult or german."
            )
            return

        # train test split
        self.dataset_orig_train, self.dataset_orig_test = self.dataset_orig.split(
            [0.7], shuffle=True
        )

        # protected features
        self.privileged_groups = [{name: 1} for name in protected_attributes_names]
        self.unprivileged_groups = [{name: 0} for name in protected_attributes_names]

        # normalize data
        self.scale_orig = StandardScaler()
        self.X_train, self.y_train = self.scale(self.dataset_orig_train)
        self.X_test, self.y_test = self.scale(self.dataset_orig_test)

    def scale(self, data: AdultDataset or GermanDataset):
        """
        Normalize data
        """
        x = self.scale_orig.fit_transform(data.features)
        y = data.labels.ravel()

        return x, y

    def get_data(self):
        """
        Returns features and labels for training and test data
        """
        return self.X_train, self.y_train, self.X_test, self.y_test

    def compute_fairness_metric(self, dataset=None):
        """
        Returns mean difference in FPR and TPR for unprivileged and privileged groups.
        """
        if dataset is None:
            dataset = self.dataset_orig_train
        metric_orig_train = BinaryLabelDatasetMetric(
            dataset,
            unprivileged_groups=self.unprivileged_groups,
            privileged_groups=self.privileged_groups,
        )

        return metric_orig_train.mean_difference()

    def get_metrics(self, test_X, test_pred):
        """
        Get fairness metric values of provided dataset
        """
        metric = ClassificationMetric(
            test_X,
            test_pred,
            unprivileged_groups=self.unprivileged_groups,
            privileged_groups=self.privileged_groups,
        )
        metric_arrs = {}

        # Statistical Parity Difference measures the difference of the above values instead of ratios, hence we
        # would like it to be close to 0.
        metric_arrs["stat_par_diff"] = metric.statistical_parity_difference()

        # Equal opportunity difference measures the ability of the classifier to accurately classify a datapoint as positive
        # regardless of the presence of the unpriviliged feature. We would like it to be close to 0. A negative value signals bias
        # towards priviliged.
        metric_arrs["equal_opp_diff"] = metric.equal_opportunity_difference()

        # Average of difference in FPR and TPR for unprivileged and privileged
        # groups. A value of 0 indicates equality of odds.
        metric_arrs["avg_odds_diff"] = metric.average_odds_difference()

        # Balanced accuracy is a general metric, not dependent on bias. We would like to have it close to 1, meaning
        # that the classifier can equally detect positive and negative classes.
        metric_arrs["bal_acc"] = (
            metric.true_positive_rate() + metric.true_negative_rate()
        ) / 2

        # We would like Disparate Impact to be close to 1. It measures the ratio between the likelihood of the class being
        # predicted as positive if we have the unpriviliged feature and the the same likelihood with the priviliged feature.
        # Values close to 0 indicate strong bias.
        metric_arrs["disp_imp"] = metric.disparate_impact()

        return metric_arrs

    def mitigate_bias(self, method="reweighing", dataset: Dataset = None):
        """
        Mitigate bias

        :param method: reweighing or debiasing
        :param dataset: dataset to mitigate bias on

        :return: dataset with bias mitigated
        """
        if dataset is None:
            dataset = self.dataset_orig_train

        dataset_transf = None

        if method == "reweighing":
            RW = Reweighing(
                unprivileged_groups=self.unprivileged_groups,
                privileged_groups=self.privileged_groups,
            )
            print("Mitigation: Reweighing complete")
            dataset_transf = RW.fit_transform(self.dataset_orig_train)

        elif method == "a-debaising":
            # AD = AdversarialDebiasing(unprivileged_groups=self.unprivileged_groups,
            #         privileged_groups=self.privileged_groups)
            # dataset_transf = AD.fit_transform(self.dataset_orig_train)
            # print("Mitigation: Adversarial Debiasing complete")
            print("Method Not Implemented: No mitigation done")
        else:
            print("Method Not Implemented: No mitigation done")

        return dataset_transf
