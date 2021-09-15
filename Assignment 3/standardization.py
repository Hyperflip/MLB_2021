from scipy import stats
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from util import plot_distribution

if __name__ == '__main__':
    iris = load_iris()
    data = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                        columns=iris['feature_names'] + ['target'])
    X = data[:-1]

    # plot normal distribution before standardization
    sepal_lengths = data.iloc[:, 0]
    plot_distribution(sepal_lengths, 'sepal length distribution BEFORE standardization')

    for col_name, col_data in X.iteritems():
        mu, sigma = stats.norm.fit(col_data)
        X[col_name] = col_data.apply(lambda x: (x - mu) / sigma)

    # plot normal distribution after standardization
    sepal_lengths = data.iloc[:, 0]
    plot_distribution(sepal_lengths, 'sepal length distribution AFTER standardization')

    print(data)
