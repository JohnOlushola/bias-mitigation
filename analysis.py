import matplotlib.pyplot as plt
import pandas as pd

from dataset import Dataset
from models import LogisticRegressionClassifier
import numpy as np

# Plot configuration
plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.figsize"] = (6, 6)
plt.rcParams["lines.markersize"] = 16
plt.rcParams["lines.color"] = "blue"
plt.rcParams["lines.color"] = "black"
plt.style.use("seaborn")


class Analysis:
    def __init__(
        self,
        dataset: Dataset,
        C_values=[],
        solver="liblinear",
        params={},
        include_delta=False,
        mitigated=False,
    ):
        self.dataset = dataset
        self.C_values = C_values
        self.solver = solver
        self.mitigated = mitigated
        self.params = params
        self.results = []
        self.df: pd.DataFrame = None
        self.best_accuracy = None
        self.best_fairness = None
        self.best_accuracy_fairness = None
        self.include_delta = include_delta

    def  _closest_value(self, input_list, input_value, return_index=True):
        """
        Returns the index or value of the closest value to input_value in the list

        :param input_list: list of values
        :param input_value: value to find closest to
        :param return_index: if True, return index of closest value, else return value

        :return: index or value of closest value
        """
        arr = np.asarray(input_list)
        print(arr)
        i = (np.abs(arr - input_value)).argmin()
        return i if return_index else arr[i]
        

    def run(self, verbose=False):
        """
        Run analysis on the dataset

        :param verbose: if True, print results

        :return: dictionary of results
        """
        models = [
            LogisticRegressionClassifier(
                verbose=verbose,
                dataset=self.dataset,
                solver=self.solver,
                params={"C": value, **self.params},
                privileged_groups=self.dataset.privileged_groups,
                unprivileged_groups=self.dataset.unprivileged_groups,
            )
            for value in self.C_values
        ]

        self.results = [
            (
                model.model.C,
                *model.run(
                    self.dataset,
                    mitigated=self.mitigated,
                    include_delta=self.include_delta,
                ),
            )
            for model in models
        ]

        columns = (
            ["model", "accuracy", "accuracy_std", "fairness", "accuracy+fairness"]
            if self.include_delta
            else ["model", "accuracy", "accuracy_std", "fairness"]
        )

        # display results
        self.df = pd.DataFrame(self.results, columns=columns)

        # best accuracy i.e. accuracy closest to 1
        self.best_accuracy = self.df["accuracy"].idxmax()

        # best fairness i.e. fairness closest to 0
        self.best_fairness = self.df['fairness'].sub(0).abs().idxmin()

        # best accuracy+fairness i.e. lowest accuracy+fairness
        if self.include_delta:
            self.best_accuracy_fairness = self.df["accuracy+fairness"].idxmin()

        print("Best accuracy: " + str(self.best_accuracy))
        print("Best fairness: " + str(self.best_fairness))
        print("Best accuracy+fairness: " + str(self.best_accuracy_fairness))

        return {
            "data_frame": self.df,
            "best_accuracy": self.best_accuracy,
            "best_fairness": self.best_fairness,
            "best_accuracy_fairness": self.best_accuracy_fairness,
            "results": self.results,
            "models": models,
            "best_accuracy_model": models[self.best_accuracy],
            "best_fairness_model": models[self.best_fairness],
            "best_accuracy_fairness_model": models[self.best_accuracy_fairness]
            if self.include_delta
            else None,
        }

    def plot_results(self, type=None):
        """
        Plots results of analysis done

        :param type: if None, plot all results, else plot only the specified type
        """

        df = self.df

        x = df["model"]
        y_accuracy = df["accuracy"]
        y_fairness = df["fairness"]
        if self.include_delta:
            y_accuracy_fairness = df["accuracy+fairness"]

        xi = range(len(x))

        if type == "accuracy":
            # plot accuracy
            plt.plot(xi, y_accuracy, color="black")
            plt.xlabel("C")
            plt.ylabel("Accuracy")
            plt.legend(["Accuracy", "Best Accuracy"])
            plt.title("Accuracy vs C")
        elif type == "fairness":
            # plot fairness
            plt.plot(xi, y_fairness, color="black")
            plt.xlabel("C")
            plt.ylabel("Fairness")
            plt.legend(["Fairness", "Best Fairness"])
            plt.title("Fairness vs C")
        elif type == "accuracy+fairness":
            # plot accuracy+fairness
            plt.plot(xi, y_accuracy_fairness, color="black")
            plt.xlabel("C")
            plt.ylabel("Accuracy+Fairness")
            plt.legend(["Accuracy+Fairness", "Best Accuracy+Fairness"])
            plt.title("Accuracy+Fairness vs C")
        else:
            # plot accuracy and fairness
            fig, ax1 = plt.subplots()

            # accuracy
            ax1.plot(xi, y_accuracy, markersize=6, color='black')
            ax1.set_ylabel('Accuracy', color='black')
            ax1.tick_params(axis='y', labelcolor='black')

            # fairness
            ax2 = ax1.twinx()
            ax2.plot(xi, y_fairness, markersize=6, color='blue')
            ax2.set_ylabel('Fairness', color='blue')
            ax2.tick_params(axis='y', labelcolor='blue')
            ax2.grid(False, axis='y')

            plt.xlabel("C")

        # x-ticks
        plt.xticks(xi, x)

        return plt.show()
