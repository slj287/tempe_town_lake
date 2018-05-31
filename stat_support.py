
"""stat_support.py -- Functions to calculate, locate, and plot outliers.

"""

import matplotlib.pyplot as plt
import numpy as np


# The data set is so large, the number of "outliers" is unreasonably large.
# So we choose a "scale" for the inter-quartile range that requires outliers
# to be more than just a few SDs from the mean.
#
# As a simple experiment, outliers were required to be more SDs from
# the mean than the traditional 3 (1.5).
#
#default_iqr_scale = 5.5
default_iqr_scale = 1.5

def outlier_bounds(values, iqr_scale=default_iqr_scale):

    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    lower_bound = q1 - iqr_scale * (q3 - q1)
    upper_bound = q3 + iqr_scale * (q3 - q1)
    print("checking %d values " % (len(values)) +
          "(q1=%.1f, q3=%.1f, iqr_scale=%.1f): " % (q1, q3, iqr_scale) +
          "<= %f or >= %f" % (lower_bound, upper_bound))
    return lower_bound, upper_bound


def find_outlier_indices(values, iqr_scale=default_iqr_scale):
    lower_bound, upper_bound = outlier_bounds(values, iqr_scale)
    indices_of_outliers = []
    for ind, value in enumerate(values):
        if (value <= lower_bound) or (value >= upper_bound):
            indices_of_outliers.append(ind)
    return indices_of_outliers


def find_outliers(values, iqr_scale=default_iqr_scale):
    """values should be a Pandas Series

    """
    lower_bound, upper_bound = outlier_bounds(values)
    return values[(values <= lower_bound) | (values >= upper_bound)]


def plot_outliers(dist, iqr_scale=default_iqr_scale):
    """dist is an array indexed by position"""
    dist_indices = find_outlier_indices(dist, iqr_scale)
    #print(dist)
    print("outlier count = %d" % (len(dist_indices)))
    fig = plt.figure()
    ax = fig.add_subplot(111)               # 1x1 grid, first subplot
    ax.plot(dist, 'b-', label='distances')
    ax.plot(dist[dist_indices],
         'ro',
        markersize = 7,
        label='outliers')
    ax.legend(loc='best')
    plt.show()
