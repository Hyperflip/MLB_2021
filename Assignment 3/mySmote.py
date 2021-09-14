from collections import Counter
from operator import itemgetter
import random
from scipy import stats
import numpy as np
from util import distance_euclid


class MySmote:
    X = []
    y = []
    X_minor = []
    minority_class = 0
    majority_count = 0
    k = 0

    def __init__(self, k):
        self.k = k

    def fit_transform(self, X, y):
        c = Counter(y)
        min_key, min_count = min(c.items(), key=itemgetter(1))
        _, max_count = max(c.items(), key=itemgetter(1))
        self.X = X
        self.y = y
        self.X_minor = X[[i for i in range(len(X)) if y[i] == min_key]]
        self.minority_class = min_key
        self.majority_count = max_count

        self._transform()
        return self.X, self.y

    def _transform(self):
        for i in range(0, self.majority_count - len(self.X_minor)):
            # get random sample
            rand_sample = self.X_minor[random.randint(0, len(self.X_minor) - 1)]

            # calculate closest k neighbouring samples
            closest_samples = []
            for e in self.X_minor:
                if all(np.equal(e, rand_sample)):
                    continue
                closest_samples.append((distance_euclid(rand_sample, e, 2), e))
            closest_samples = sorted(closest_samples)[0:self.k]

            # pick random from closest samples
            rand_closest = closest_samples[random.randint(0, self.k - 1)][1]

            # from 100 x and y values between the sample and random neighbour, calculate normal distribution
            x_values = np.linspace(rand_sample[0], rand_closest[0], 100)
            y_values = np.linspace(rand_sample[1], rand_closest[1], 100)
            mu_x, std_x = stats.norm.fit(x_values)
            mu_y, std_y = stats.norm.fit(y_values)
            # choose random x and y values for new sample from normal distribution
            new_x = np.random.normal(mu_x, std_x)
            new_y = np.random.normal(mu_y, std_y)

            # append new sample to minority
            new_sample = np.array([new_x, new_y])
            self.X = np.concatenate((self.X, [new_sample]), axis=0)
            self.y = np.append(self.y, self.minority_class)
