import numpy as np
import matplotlib.pyplot as plt
import seaborn
from matplotlib.ticker import MaxNLocator

seaborn.set()
plt.style.use(
    "https://raw.githubusercontent.com/NeuromatchAcademy/course-content/master/nma.mplstyle")


def plot_variance_explained(variance_explained):
    """
    Plots eigenvalues.

    Args:
    variance_explained (numpy array of floats) : Vector of variance explained
                                                 for each PC

    Returns:
    Nothing.

    """

    plt.figure()
    plt.plot(np.arange(1, len(variance_explained) + 1),
             variance_explained, '--k', marker='.')
    plt.xlabel('Number of components')
    plt.ylabel('Variance explained')

    # Force ticks to be at integer values
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.show()


def visualize_components(component1, component2, labels_index, labels, show=True):
    """
    Plots a 2D representation of the data for visualization with categories
    labelled as different colors.

    Args:
      component1 (numpy array of floats)    : Vector of component 1 scores
      component2 (numpy array of floats)    : Vector of component 2 scores
      labels_index (numpy array of floats)  : Vector corresponding to categories 
                                              indicies of samples
      labels (numpy array of strings)       : Vector corresponding to categories 
                                              names of samples

    Returns:
      Nothing.

    """

    plt.figure()
    n_labels = len(labels)  # number of labels for the data
    # Good options for color maps are:
    # cmap_name = 'plasma'
    cmap_name = 'cubehelix'
    # cmap_name = 'tab20'
    cmap = cmap = plt.cm.get_cmap(cmap_name, n_labels)
    plt.scatter(x=component1, y=component2, c=labels_index, cmap=cmap)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    formatter = plt.FuncFormatter(lambda val, loc: labels[val-1])
    plt.colorbar(ticks=np.arange(n_labels+1), format=formatter)
    plt.clim(-0.5, n_labels-0.5)
    if show:
        plt.show()


def get_variance_explained(evals):
    """
    Calculates variance explained from the eigenvalues.

    Args:
      evals (numpy array of floats) : Vector of eigenvalues

    Returns:
      (numpy array of floats)       : Vector of variance explained

    """

    # cumulatively sum the eigenvalues
    csum = np.cumsum(evals)
    # normalize by the sum of eigenvalues
    variance_explained = csum / csum[-1]

    return variance_explained
