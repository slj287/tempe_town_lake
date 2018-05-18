
"""Utility definitions and functions for analyzing winds in the
vicinity of Tempe Town Lake.

"""

from datetime import datetime
import time

import pandas as pd

months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
month_numbers = dict([(month, number)
                      for number, month in enumerate(months, 1)])

filling_began = datetime(1999,6,2)
declared_full = datetime(1999,7,14)
burst = datetime(2010,7,20)
reopened = datetime(2010, 10, 26)

def load_asos(f, index_col=1):

    df = pd.read_csv(f, comment="#", skipinitialspace=True, na_values=["M"],
                     index_col=index_col,
                     infer_datetime_format=True, parse_dates=[index_col])
    df_sans_nans = df[df.sknt.notna() & df.drct.notna()].copy()
    return df_sans_nans


def dedup_to_one_hourly_reading(df):
  """
  The input DataFrame should be indexed by the timestamp (aka "valid") field.
  It should already be stripped of NaN wind speed/direction values.
  1) Create a new column, Hourly, that has the closest hour for every reading.
  2) Create a new column, OffsetFromHour, that has the absolute value
     (in minutes) of how far from the nearest hour each reading was made.
  3) For every hour, sort readings made for the same hour in descending order
     of how close it was made to that hour, and keep only the closest reading.
  """
  df = df.copy()
  hourly = df.index.round("1H")
  df.loc[:, "Hourly"] = hourly
  df["OffsetFromHour"] = ((df.index - df["Hourly"])
                          .map(lambda td: td.total_seconds()/60)
                          .abs())
  grouped = (df.groupby("Hourly").
             apply(lambda by_hour: by_hour.sort_values(["OffsetFromHour"])).
             drop_duplicates(["Hourly"], keep="first"))
  grouped.index = grouped.Hourly
  return grouped


def dedup_readings(df, start=None, end=None):
    """Return readings grouped by hour.
    See: dedup_to_one_hourly_reading
    """
    if start is None:
        if end is None:
            df_to_dedup = df
        else:
            df_to_dedup = df[:end]
    else:
        if end is None:
            df_to_dedup = df[start:]
        else:
            df_to_dedup = df[start:end]

    print("> Label with Hourly (rounded to nearest hour)")
    # define an "Hourly" column
    hourly = df_to_dedup.index.round("1H")
    use_loc = True
    if use_loc:
        df_to_dedup.loc[:, "Hourly"] = hourly
    else:
        df_to_dedup["Hourly"] = hourly
    # and a difference between each time and its rounded value
    if False:
        df_to_dedup.index = df.index.levels[0]

    assert all(df_to_dedup.index[df_to_dedup.index.second==0])
    # (time2 - time1) => timedelta64.total_seconds()
    print("> Calculate OffsetFromHour as absolute difference from Hourly")
    t0 = time.time()
    df_to_dedup["OffsetFromHour"] = ((df_to_dedup.index - df_to_dedup["Hourly"])
                                     .map(lambda td: td.total_seconds()/60)
                                     .abs())

    t = time.time()
    print("> That took %s seconds. Added as OffsetFromHour column." % (t-t0))

    # Takes 10-15 minutes on mbai7, but cuts the number of readings by almost half
    print("> De-dup Hourly: Sort by OffsetFromHour, & keep only the one closest to Hourly")
    t0 = time.time()
    grouped = (df_to_dedup.groupby("Hourly").
               apply(lambda by_hour: by_hour.sort_values(["OffsetFromHour"])).
               drop_duplicates(["Hourly"], keep="first"))
    t = time.time()
    print("> That took %s seconds. Result saved in 'grouped'." % (t-t0))

    print("> Making Hourly the index")
    if False:
        # Fix this (2018-04-12):
        # ValueError: Length of values does not match length of index
        grouped["timestamp"] = grouped.index
        grouped.index = grouped.Hourly

    return grouped


def make_groups(source_df):
    """Take readings grouped by hour and group those by month. (???)
    """
    dfmg = source_df.groupby((source_df.index.levels[0].month))

    month_hour_dfs = {m:{} for m in months}
    for month_number, month_name in enumerate(months, 1):
        if month_number not in dfmg.groups:
            continue
        month_df = dfmg.get_group(month_number)
        dfhg = month_df.groupby((month_df.index.levels[0].hour))
        for hour_number in dfhg.groups:
            month_hour_dfs[month_name][hour_number] = dfhg.groups[hour_number]
    return month_hour_dfs


def by_hour(ddg):
    """
    ddg is a deduped_group
    """

    groupby_hour = ddg.groupby((ddg.index.hour))
    h = dict((n, groupby_hour.get_group(n)) for n in groupby_hour.groups)
    return h


def by_month(ddg):
    """
    ddg is a deduped_group
    """
    groupby_month = ddg.groupby((ddg.index.month))
    m = dict((n, groupby_month.get_group(n)) for n in groupby_month.groups)
    return m
