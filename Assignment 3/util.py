import math as mt
from numpy import where
from matplotlib import pyplot
from collections import Counter
import matplotlib.pyplot as plt
import scipy.stats as stats

def distance_euclid(p, q, dim):
    # p, d ... points, dim ... dimension
    sum_squared = 0
    for i in range(dim):
        sum_squared += (p[i] - q[i]) ** 2
    return mt.sqrt(sum_squared)

def plot_classes(X, y, title):
    # summarize the new class distribution
    counter = Counter(y)
    # scatter plot of examples by class label
    for label, _ in counter.items():
        row_ix = where(y == label)[0]
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
    pyplot.legend()
    pyplot.title(title)
    pyplot.show()

def plot_distribution(x, title):
    x = sorted(x)
    fig, ax = plt.subplots(1, 1)
    mu, sigma = stats.norm.fit(x)
    ax.plot(x, stats.norm.pdf(x, mu, sigma))
    plt.title(title)
    plt.show()