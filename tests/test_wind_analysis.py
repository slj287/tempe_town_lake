
"""
Template: ~/Dropbox/Templates/test_template.py

~/anaconda/bin/pythonw test_wind_analysis.py
   or
~/anaconda/bin/pythonw tests/test_wind_analysis.py
"""

import unittest
import logging
import logging.config
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

from data_sources import in_out_file_map


cwd = os.getcwd()
if os.path.dirname(cwd) not in sys.path:
    print("add %s to sys.path" % (cwd))
    sys.path.append(cwd)

from wind_analysis import (load_winds,
                           get_deduped_df,
                           plot_avg_by_hour, plot_monthly_avg_by_hour,
                           by_month, by_hour)


def get_period_data():

    df = load_winds("ddg.csv")
    m_before = by_month(df['1994':'1998'])
    m_after = by_month(df['2000':'2004'])
    m_after2 = by_month(df['2005':'2009'])
    m_after3 = by_month(df['2011':'2015'])
    after = dict((mn ,[hnd.sknt.mean()
                       for (hn, hnd) in by_hour(m_after[mn]).items()])
                 for mn in m_after)
    before = dict((mn, [hnd.sknt.mean()
                        for (hn, hnd) in by_hour(m_before[mn]).items()])
                  for mn in m_before)
    after2 = dict((mn, [hnd.sknt.mean()
                        for (hn, hnd) in by_hour(m_after2[mn]).items()])
                  for mn in m_after2)
    after3 = dict((mn, [hnd.sknt.mean()
                       for (hn, hnd) in by_hour(m_after3[mn]).items()])
                 for mn in m_after)
    periods = [
        ("Before (1994-1998)", before),
        ("After (2000-2004)", after),
        ("After (2005-2009)", after2),
        ("After (2011-2015)", after3),
    ]
    return periods


class Tests(unittest.TestCase):

    def test_0_getting_deduped_dfs(self):

        for tag in in_out_file_map:
            df = get_deduped_df(tag)

    def test_0_verify_no_nans(self):

        for tag in in_out_file_map:
            df = get_deduped_df(tag)
            self.assertFalse(df.drct.isna().any())
            self.assertFalse(df.sknt.isna().any())

    def test_1(self):

        periods = get_period_data()
        print("Q in the plot window to Quit")
        plot_avg_by_hour(periods, 1, 0, 0)
        plt.show()

    def test_2(self):

        periods = get_period_data()
        print("Q in the plot window to Quit")
        plot_monthly_avg_by_hour(periods)
        plt.show()


if __name__ == "__main__":

    unittest.main()
