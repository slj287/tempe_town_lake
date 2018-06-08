
"""stat_support.py -- Functions to calculate, locate, and plot outliers.

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def enumerate_df(df):

    for i,v in df.iterrows():
        yield (i,v)


def find_outlier_indices(values, iqr_scale=default_iqr_scale):

    lower_bound, upper_bound = outlier_bounds(values, iqr_scale)
    indices_of_outliers = []
    if values.__class__ == pd.DataFrame:
        enumerator = enumerate_df
    else:
        enumerator = enumerate
    for ind, value in enumerator(values):
        if (value <= lower_bound) or (value >= upper_bound):
            indices_of_outliers.append(ind)
    return indices_of_outliers


def find_outliers(values, iqr_scale=default_iqr_scale):
    """values should be a Pandas Series

    """
    lower_bound, upper_bound = outlier_bounds(values, iqr_scale)
    return values[(values <= lower_bound) | (values >= upper_bound)]


def plot_marked(dist, marks=None, ax=None):

    if marks is None:
        print("mark count = 0")
    else:
        print("mark count = %d" % (len(marks)))
    print("point count: %d" % (len(dist)))
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)               # 1x1 grid, first subplot
        show = True
    else:
        print("received ax: %s" % (ax))
        show = False
    #print("dist = %s" % (dist))
    ax.plot(dist, 'b-', label='speeds')
    if marks is not None and len(marks) > 0:
        ax.plot(dist.loc[marks.index],
                'ro',
                markersize = 7)
    ax.legend(loc='best')
    if show:
        plt.show()


def plot_outliers(dist, iqr_scale=default_iqr_scale):
    """dist is an array indexed by position"""
    dist_indices = find_outlier_indices(dist, iqr_scale)
    print("outlier count = %d" % (len(dist_indices)))
    _, upper_bound = outlier_bounds(dist, iqr_scale)
    print("upper bound = %s" % (upper_bound))
    fig = plt.figure()
    ax = fig.add_subplot(111)               # 1x1 grid, first subplot
    ax.plot(dist, 'b-', label='speeds')
    outlier_points = dist[dist_indices] if len(dist_indices) > 0 else []
    ax.plot(outlier_points,
         'ro',
        markersize = 7,
        label='outliers')
    ax.legend(loc='best')
    ax.set_title("outliers >%.1f, iqr_scale=%.1f" % (upper_bound, iqr_scale))
    plt.show()
