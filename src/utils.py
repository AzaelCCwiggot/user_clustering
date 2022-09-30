# https://stats.stackexchange.com/questions/173636/clustering-of-very-skewed-count-data-any-suggestions-to-go-about-transform-et

import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.stats import chi2_contingency


def phi_distance(x, y):
    contingency_table = np.array([x,y])
    contingency_table = contingency_table[:, ~np.all(contingency_table == 0, axis=0)] # Remove empty column
    chi2 = chi2_contingency(contingency_table, correction=False)[0]
    return np.sqrt(chi2/np.sum(contingency_table))


if __name__ == "__main__":
    X = np.array([[12,5,0], [3,4,0], [2,21,0], [4,8,1]])
    print(pairwise_distances(X, metric=phi_distance))