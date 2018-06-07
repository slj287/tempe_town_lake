
"""Utility definitions and functions for analyzing winds in the
vicinity of Tempe Town Lake.

mpin         - Month Period Index, from 0 (=Dec), where Jan = 1
mpins        - The Month Period Index as a Series
month_index  - np.array(January .. December)
month3_index - np.array(Jan .. Dec)
"""

from datetime import datetime
import time

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd

from data_sources import in_out_file_map
from stat_support import outlier_bounds


__version__ = 4

mpin = pd.PeriodIndex(start='Dec', periods=13, freq="M").strftime("%b")
mpins = pd.Series(list(range(0, 13)), index=mpin)
month_index = pd.PeriodIndex(start='Jan', periods=12, freq="M").strftime("%B")
month3_index = pd.PeriodIndex(start='Jan', periods=12, freq="M").strftime("%b")

months = list(month3_index)  # Jan .. Dec
month_numbers = dict([(month, number)
                      for number, month in enumerate(months, 1)])
month_names = list(month_index)

filling_began = datetime(1999,6,2)
declared_full = datetime(1999,7,14)
burst = datetime(2010,7,20)
reopened = datetime(2010, 10, 26)

default_station_sym = "PHX"


_saved_raw_asos = dict()
def get_asos_df(station_sym=default_station_sym, copy=True):
    """Note: Only set copy=False if the return value is going to be kept
    read-only.

    """
    global _saved_raw_asos

    if station_sym in _saved_raw_asos:
        if copy:
            return _saved_raw_asos[station_sym].copy()
        else:
            return _saved_raw_asos[station_sym]
    infile, idx, _ = in_out_file_map[station_sym]
    print("> Load raw readings from %s" % (infile))
    df0 = load_asos(infile, index_col=idx)
    _saved_raw_asos[station_sym] = df0
    if copy:
        return df0.copy()
    else:
        return df0


_saved_winds = dict()
def get_starting_df(station_sym=None):

    _station_sym = station_sym or default_station_sym
    if _station_sym in _saved_winds:
        return _saved_winds[_station_sym].copy()
    df0 = get_asos_df(_station_sym)
    df1 = narrow_asos_df_to_winds(df0)
    _saved_winds[_station_sym] = df1
    return df1.copy()


def load_asos(f, index_col=1):

    df = pd.read_csv(f, comment="#", skipinitialspace=True, na_values=["M"],
                     index_col=index_col,
                     infer_datetime_format=True, parse_dates=[index_col])
    df_sans_nans = df[df.sknt.notna() & df.drct.notna()].copy()
    return df_sans_nans


def narrow_asos_df_to_winds(df):
    """In pulling from generate_for_station, we dropped a hardcoded switch
    in case pd.DatetimeIndex is already indexable. The code here seems
    to work either way.

    """

    df["timestamp"] = pd.DatetimeIndex(df.index)

    print(">!!! Validate wind speed against METAR (TBI)")
    if False:
        df["sknt_metar"] = df.metar.map()
    print("> Narrowing to timestamp, drct, and sknt (+ dropping NaNs)")
    wind_df = df[df.sknt.notna() & df.drct.notna()][['timestamp', 'drct', 'sknt']]
    return wind_df


def narrow_asos_df_to_valid_hourly(df):

    wind_df = narrow_asos_df_to_winds(df)

    print("> Dedup readings")
    deduped_group = dedup_readings(wind_df, cleanup=True)

    print("To get NaNs for hours without readings, wind_df.resample(\"H\")?")
    return deduped_group


def load_winds(f):

    deduped_group = pd.read_csv(f, index_col=0,
                                infer_datetime_format=True,
                                parse_dates=[0])
    return deduped_group


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


def dedup_readings(df, start=None, end=None, cleanup=True):
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

    if cleanup:
        if "Hourly" in grouped.columns:
            grouped.index = grouped.Hourly
        cols_to_drop = (set(grouped.columns)
                        .intersection(set(["Hourly", "Hourly.1",
                                           "OffsetFromHour"])))
        if cols_to_drop:
            grouped.drop(cols_to_drop, axis=1, inplace=True)

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


def month_index_generator():

    nr = 4
    nc = 3
    # 012 345 678 9ab puts 0 in the LL corner
    # we want 9ab 678 345 012, which is 12 - (nc*(1+(mn//nc))) + (mn%nc)
    for mn in range(12):
        b = (nc*(1+(mn//nc)))
        c = (mn%nc)
        fn = (12 - b + c)
        yield fn


def plot_avg_by_hour(period_list, month, i, j):

    figwidth = 1.5
    figheight = 1.3

    month_n = month - 1
    for_month = dict([(k, v[month_n+1]) for (k,v) in period_list])

    mdf = pd.DataFrame(for_month,
                       columns=for_month.keys())
    a = mdf.plot()
    if i != 0:
        a.set_xticklabels([])
    a.set_ylim(-0.5,11.0)
    if i == 0:
        x = a.get_xaxis()
        #print("%s" % (a.get_xticklabels()))
        a.set_xticklabels([0, 6, 12, 18, 24])

    x0 = 10.0
    y0 = 8 * figheight * .95
    print("Title text @ (%.f, %f)" % (x0, y0))
    bb = a.get_window_extent()
    print("width, height = %f, %f" % (bb.width, bb.height))
    a.text(x0, y0, "%s" % (month_names[month_n]), fontsize="xx-large")

    plt.legend(bbox_to_anchor=(0, -0.1), loc=2, borderaxespad=0.)
    return a


def plot_monthly_avg_by_hour(period_list):
    """
    period_list - A list of (period_name, avgs_by_hour) tuples.

    Example:
      df = load_winds(from_file)
      m_before = by_month(df['1994':'1998'])
      m_after = by_month(df['2000':'2004'])
      by_month_before = dict((mn,
                              [hnd.sknt.mean()
                              for (hn, hnd) in
                              by_hour(m_before[mn]).items()])
                             for mn in m_before)
      by_hour_each_month_before = pd.DataFrame(by_month_before)
      by_month_after = dict((mn,
                             [hnd.sknt.mean()
                             for (hn, hnd) in
                             by_hour(m_after[mn]).items()])
                            for mn in m_after)
      by_hour_each_month_after = pd.DataFrame(by_month_after)

     plot_monthly_avg_by_hour([("Before", before),
                               ("After", after)])

    """
    figwidth = 1.5
    figheight = 1.3
    Nr = 4
    Nc = 3
    w = (figwidth / Nc) * 0.8
    h = (figheight / Nr) * 2 / 3

    fig = plt.figure()
    figtitle = 'Mean wind speed (kts.) by hour of day for each month'
    tx = (figwidth / 2) * .80
    ty = figheight * .95
    t = fig.text(tx, ty, figtitle,
                 horizontalalignment='center', fontproperties=FontProperties(size=16))
    ax = []
    month_index = month_index_generator()
    for i in range(Nr):
        for j in range(Nc):
            pos = [0.075 + j*1.1*w, 0.18 + i*1.2*h, w, h]
            month_n = next(month_index)
            a = fig.add_axes(pos)
            if i != 0:
                a.set_xticklabels([])
            if j != 0:
                a.set_yticklabels([])
            for_month = dict([(k, v[month_n+1]) for (k,v) in period_list])
            bva = pd.DataFrame(for_month,
                               columns=for_month.keys())
            a.set_ylim(-0.5,12.0)
            if i == 0:
                x = a.get_xaxis()
                a.set_xticks([0, 6, 12, 18, 24])

            x0 = 8.0
            y0 = 7 * figheight * .95
            a.text(x0, y0, "%s" % (month_names[month_n]), fontsize="large")

            periods_before = len([x for x in bva.columns
                                  if x.lower().startswith("before")])
            periods_after = len([x for x in bva.columns
                                  if x.lower().startswith("after")])
            a.plot(bva)
            ax.append(a)
    ax[-1].legend(bva.columns, bbox_to_anchor=(.45, 1.5), loc=2, borderaxespad=0.)
    return ax


def find_extreme_outliers(df):

    lower, upper = outlier_bounds(df.sknt, iqr_scale=10)
    df1, sc = annotate_abnormal_speeds(df)
    o = df[(df.ratio < .4) & (df.sknt > upper)]
    n4 = len(o)
    n3 = len(df[(df.ratio < .3) & (df.sknt > upper)])
    n2 = len(df[(df.ratio < .2) & (df.sknt > upper)])
    n1 = len(df[(df.ratio < .1) & (df.sknt > upper)])
    # a ratio of .2 is almost as if to say the log(sknt) is within a factor of 2 of the others
    print("> method_1: Number of points with ratio < (.4, .3, .2, .1): " +
          "%d, %d, %d, %d" % (n4, n3, n2, n1))
    return o


def annotate_abnormal_speeds(odf):

    print("> Find abnormal speeds")
    df = odf
    print(">   prev")
    df.loc[:,"prev"] = df.sknt.shift()
    print(">   next")
    df.loc[:,"next"] = df.sknt.shift(-1)
    print(">   ratio")
    df.loc[:,"ratio"] = (df.prev + df.next) / (df.sknt + df.prev + df.next)
    # restrict first to significantly (non-zero) changes:
    print(">   generate sc")
    sc = df.loc[df["ratio"].abs() < .1]
    return df, sc
