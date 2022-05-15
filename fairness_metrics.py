from aif360.metrics import ClassificationMetric


class Fairness:
    def __init__(self, X, y, unprivileged_groups, privileged_groups):
        self.metric = ClassificationMetric(X, y, unprivileged_groups, privileged_groups)

    def statistical_parity_difference(self):
        """
        The closer this value is to 0 means each member of both groups is equally likely to be selected.
        """
        return self.metric.statistical_parity_difference()

    def eq_opp_diff(self):
        """
        Returns true positive rate difference between privileged and unprivileged groups.
        The closer this value is to 0 the fairer the model
        """
        return self.metric.equal_opportunity_difference()

    def avg_odds_diff(self):
        """
        Returns average in the difference between TPR and FPR for privileged and unprivileged groups.
        0 means both group have equal odds
        """
        return self.metric.average_odds_difference()

    def true_positive_rate(self):
        """
        Returns true positive rate
        """
        return self.metric.true_positive_rate()

    def true_negative_rate(self):
        """
        Returns true negative rate
        """
        return self.metric.true_negative_rate()

    def bal_acc(self): 
        """
        Returns 1 if the classifier can equally detect positive and negative classes. 
        It is not dependent on bias.
        """
        return (self.true_positive_rate() + self.true_negative_rate()) / 2

    def disp_imp(self):
        """
        Returns the ratio of the likelihood of the class being predicted as positive 
        if we have the unpriviliged feature and the same likelihood with the priviliged feature.
        """
        return self.metric.disparate_impact()

    def get_metrics(self):
        """
        Returns all metric in a dictionary
        """
        return {
            "stat_par_diff": self.statistical_parity_difference(),
            "eq_opp_diff": self.eq_opp_diff(),
            "avg_odds_diff": self.avg_odds_diff(),
            "bal_acc": self.bal_acc(),
            "disp_imp": self.disp_imp(),
        }
