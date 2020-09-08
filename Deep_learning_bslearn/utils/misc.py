import progressbar
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import numpy as np

from From_Scratch_sample_implementations.Deep_learning_bslearn.utils.data_metrics import calculate_covariance_matrix
from From_Scratch_sample_implementations.Deep_learning_bslearn.utils.data_metrics import calculate_correlation_matrix
from From_Scratch_sample_implementations.Deep_learning_bslearn.utils.data_manipulation import Standardize



bar_widgets = ['Training: ', progressbar.Percentage(), ' ', progressbar.Bar(marker="-", left="[",
                                                                            right="]")
               , ' ', progressbar.ETA()]


class Plot():

    def __init__(self):
        self.cmap = plt.get_cmap('viridis')



    def _transform(self, X, dim): # PCA
        covariance = calculate_covariance_matrix(X)
        eigenvalues, eigenvectors = np.linalg.eig(covariance)

        #sort eigenvalues and vectors by largest to smallest

        idx = eigenvalues.argsort()[::-1] #argsort returns indices for sorting
        eigenvalues = eigenvalues[idx][:dim]

        eigenvectors = np.atleast_1d(eigenvectors[:,:idx])[:,:dim]

        #Project the vectors onto principal components

        X_PCA = X.dot(eigenvectors)

        return X_PCA

    def plot_regression(self, lines, title, axis_labels=None, mse=None, scatter=None,
                        legend={"type": "lines", "loc": "lower right"}):

        if scatter:
            scatter_plots = scatter_labels = []
            for s in scatter:
                scatter_plots += [plt.scatter(s["x"], s["y"], color=s["color"], s=s["size"])]
                scatter_labels += [s["label"]]
            scatter_plots = tuple(scatter_plots)
            scatter_labels = tuple(scatter_labels)

        for l in lines:
            li = plt.plot(l["x"], l["y"], color=s["color"], linewidth=l["width"], label=l["label"])

        if mse:
            plt.suptitle(title)
            plt.title("MSE: %.2f" % mse, fontsize=10)
        else:
            plt.title(title)

        if axis_labels:
            plt.xlabel(axis_labels["x"])
            plt.ylabel(axis_labels["y"])

        if legend["type"] == "lines":
            plt.legend(loc="lower_left")
        elif legend["type"] == "scatter" and scatter:
            plt.legend(scatter_plots, scatter_labels, loc=legend["loc"])

        plt.show()


