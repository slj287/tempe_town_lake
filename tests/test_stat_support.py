
"""
Template: ~/Dropbox/Templates/test_template.py

~/anaconda/bin/pythonw test_stat_support.py
   or
~/anaconda/bin/pythonw tests/test_stat_support.py
"""

import unittest
import logging
import logging.config
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

cwd = os.getcwd()
if os.path.dirname(cwd) not in sys.path:
    print("add %s to sys.path" % (cwd))
    sys.path.append(cwd)

from data_sources import in_out_file_map
from stat_support import (find_outlier_indices,
                          find_outliers,
                          plot_outliers)
from wind_analysis import get_starting_df, load_winds


class Tests(unittest.TestCase):

    def test_foo(self):

        df = load_winds("ddg.csv")
        print("Q in the plot window to Quit")
        plot_outliers(df.sknt, iqr_scale=5.5)

    def test_outlier_dropping_foi_source(self):

        df = get_starting_df("FFZ")
        indices = find_outlier_indices(df.sknt, iqr_scale=100)
        sdf = df.drop(indices)
        print("Q in the plot window to Quit")
        plot_outliers(sdf.sknt, iqr_scale=60)

    def test_outlier_dropping_fo_source(self):

        df = get_starting_df("FFZ")
        print("Q in the plot window to Quit")
        o = find_outliers(df.sknt, iqr_scale=100)
        sdf = df.drop(o.index)
        plot_outliers(sdf.sknt, iqr_scale=60)

    def test_plotting_empty_outlier_set(self):

        df = get_starting_df("FFZ")
        print("Q in the plot window to Quit")
        plot_outliers(df.sknt, iqr_scale=200)

if __name__ == "__main__":

    unittest.main()
