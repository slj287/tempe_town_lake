
"""
Experiments to understand filterability of outliers, e.g. at SDL & FFZ

TODO: Check Williams Gateway quality
"""

import unittest

import matplotlib.pyplot as plt
import pandas as pd

from stat_support import (find_outliers, find_outlier_indices,
                          outlier_bounds, plot_marked, plot_outliers)
from wind_analysis import (load_asos, narrow_asos_df_to_winds,
                           find_extreme_outliers, annotate_abnormal_speeds,
                           get_asos_df, get_starting_df)

#default_iqr_scale = 1.5
#default_iqr_scale = 5.5
default_iqr_scale = 100


def doit(station_sym=None, iqr_scale=None, df=None):

    _df = get_starting_df(station_sym) if df is None else df
    _iqr_scale = iqr_scale or default_iqr_scale
    plot_outliers(_df.sknt, iqr_scale=_iqr_scale)
    o_indices = find_outlier_indices(_df.sknt, _iqr_scale)
    print("n_outliers = %d" % (len(o_indices)))
    o = find_outliers(_df.sknt, _iqr_scale)
    return _df.loc[o.index]

def restrict_to_days_around_points(df, points):

    rdf = df.loc[df.index.map(lambda d: d.date() in points.index.date)]
    return rdf

def examine_days_around_outliers(df, points):

    print(">   Restrict to days around points")
    rdf = restrict_to_days_around_points(df, points)
    print(">   prev")
    rdf.loc[:,"prev"] = rdf.sknt.shift()
    print(">   next")
    rdf.loc[:,"next"] = rdf.sknt.shift(-1)
    if False:
        print(">   diff")
        rdf.loc[:,"diff"] = rdf.sknt.diff()
    print(">   ratio")
    rdf.loc[:,"ratio"] = (rdf.prev + rdf.next) / (rdf.sknt + rdf.prev + rdf.next)
    # restrict first to significantly (non-zero) changes:
    print(">   generate sc")
    sc = rdf.loc[rdf["ratio"].abs() < .1]
    return rdf, sc


def graph_0(df):
    print("max(df.sknt) = %f" % max(df.sknt))
    odf = doit(iqr_scale=80, df=df)
    o = find_outliers(df.sknt, 80)
    df.loc[o.index].head()
    sdf = df.drop(o.index)
    print("max(sdf.sknt) = %f" % max(sdf.sknt))


def outlier_detection_method_0(df):

    df1, sc = annotate_abnormal_speeds(df)
    o = df[df.ratio < .4]
    n4 = len(o)
    n3 = len(df[df.ratio < .3])
    n2 = len(df[df.ratio < .2])
    n1 = len(df[df.ratio < .1])
    # a ratio of .2 is almost as if to say the log(sknt) is within a factor of 2 of the others
    print("> method 0: Number of points with ratio < (.4, .3, .2, .1): " +
          "%d, %d, %d, %d" % (n4, n3, n2, n1))
    return o


def outlier_detection_method_1(df):
    """Now part of wind_analysis as find_extreme_outliers
    """
    return find_extreme_outliers(df)


def compare_outlier_methods(df=None):

    df = get_starting_df(df or "FFZ")

    o = find_outliers(df.sknt, 1.5)
    print("> Number of outlier points: %d" % (len(o)))
    plot_marked(df.sknt, o)
    o1 = find_outliers(df.sknt, 60)
    print("> Number of outlier points: %d" % (len(o1)))
    plot_marked(df.sknt, o1)
    o2 = outlier_detection_method_0(df)
    print("Dropping %d o2 points from df" % (len(o2)))
    df2 = df.drop(o2.index)
    plot_marked(df.sknt, o2)
    o3 = outlier_detection_method_1(df)
    print("Dropping %d o3 points from df" % (len(o3)))
    print(o3.sknt.describe())
    df3 = df.drop(o3.index)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    plot_marked(df.sknt, o3, axes[0])
    plot_marked(df3.sknt, None, axes[1])


def com_2(df=None):

    df = get_starting_df(df or "FFZ")
    o3 = outlier_detection_method_1(df)
    print("Dropping %d o3 points from df" % (len(o3)))
    print(o3.sknt.describe())
    print("point count before: %d" % (len(df)))
    df3 = df.drop(o3.index)
    print("point count after: %d" % (len(df3)))
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    plot_marked(df.sknt, o3, axes[0])
    plot_marked(df3.sknt, None, axes[1])
    fig.subplots_adjust()
    plt.show()



def autocorr_compares():

    raw_dfs = dict()
    for station in ["PHX", "SDL", "FFZ"]:
        raw_dfs[station] = get_asos_df(station)
    print("Overall autocorr for stations:")
    for station in ["PHX", "SDL", "FFZ"]:
        raw_df = raw_dfs[station]
        print("  %s: %.2f" % (station, raw_df.sknt.autocorr()))
        if station == "SDL":
            print("  %s (-2001): %.2f" % (station, raw_df[:"2001"].sknt.autocorr()))
            print("  %s (2002-): %.2f" % (station, raw_df["2002":].sknt.autocorr()))


def autocorr_file(f):
    df = load_asos(f, index_col=1)
    print(" %s: %.2f" % (f, df.sknt.autocorr()))


class Tests(unittest.TestCase):

    def test_com_2_0_ffz(self):

        com_2("FFZ")

    def test_com_2_1_phx(self):

        com_2("PHX")

    def test_com_2_2_sdl(self):

        com_2("SDL")


if __name__ == "__main__":

    unittest.main()
