
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm, exponweib, chisquare
from numpy import linspace
from pylab import plot,show,hist,figure,title

from wind_analysis import (dedup_readings, load_asos, make_groups,
                           by_hour, by_month,
                           month_index, mpin, mpins)

filling_began = datetime(1999,6,2)
declared_full = datetime(1999,7,14)
burst = datetime(2010,7,20)
reopened = datetime(2010, 10, 26)

take_all = True
indexable_timestamp = True

plt.style.use('seaborn') # pretty matplotlib plots


print("> Load raw readings")
df = load_asos("phx_through_2017-12-20-23:55.csv", index_col=0)

if indexable_timestamp:
    df["timestamp"] = pd.DatetimeIndex(df.index)
else:
    df["timestamp"] = df.index

phx_df = df[['timestamp', 'drct', 'sknt']]

load_from_file = "ddg-2018-04-11.csv"
if take_all:
    if True and load_from_file:
        print("> Load pre-deduped-readings")
        deduped_group = pd.read_csv(load_from_file, index_col=0,
                                    infer_datetime_format=True, parse_dates=[0])
    else:
        print("> Dedup readings")
        deduped_group = dedup_readings(phx_df)
else:
    dtd = phx_df[-1000:]
    dtd.loc["Hourly1"] = dtd.index.round("1H")
    deduped_group = dedup_readings(phx_df, start=-1000)

if "Hourly" in deduped_group.columns:
    deduped_group.index = deduped_group.Hourly

cols_to_drop = (set(deduped_group.columns)
                .intersection(set(["Hourly", "Hourly.1", "OffsetFromHour"])))
if cols_to_drop:
    deduped_group.drop(cols_to_drop, axis=1, inplace=True)


h = by_hour(deduped_group)
m = by_month(deduped_group)

# https://codereview.stackexchange.com/questions/96761/chi-square-independence-test-for-two-pandas-df-columns

def test_fit(y):

    mu, std = norm.fit(y)

    x_continuous = linspace(0, y.max())
    pdf_fitted = norm.pdf(x_continuous, loc=mu, scale=std)

    x_discrete = linspace(0, y.max(), num=int(y.max())+1)
    pdf_discrete = norm.pdf(x_discrete, loc=mu, scale=std)

    y_counts = y.value_counts()
    chsq, p = chisquare(y_counts.reindex(x_discrete, fill_value=0.),
                        pdf_discrete)

    title("Normal distribution: $\mu$ = %.2f,  std = %.2f, " % (mu, std) +
          "$\chi^2$ = %.2f, p = %.2f" % (chsq, p))
    plot(x_continuous, pdf_fitted, 'r-')
    plot(x_discrete, pdf_discrete, 'o')
    hist(y, normed=1, alpha=.3, bins=int(y.max()))

    show()
    return mu, std


def test_fit_weibull(y):
    """The PDF really doesn't look like the data. Might the parameters be wrong?
    """
    a, b, loc, scale = exponweib.fit(y)
    x = linspace(0, y.max())
    pdf_fitted = exponweib.pdf(x, a, b, loc, scale)

    title("Weibull distribution: (a=%.2f,b=%.2f) loc = %.2f,  scale = %.2f" % (a, b, loc, scale))
    plot(x, pdf_fitted, 'r-')
    hist(y, normed=1, alpha=.3, bins=int(y.max()))
    show()
    return a, b, loc, scale

aug = m[8]
aug_h = by_hour(aug)
#aug_12_post_fill = aug_h[12][aug_h[12].index.year.isin(range(1999,2010))]
test_fit(aug_h[12][aug_h[12].index.year.isin(range(1999,2010)) & aug_h[12].drct.isin(range(45,135))].sknt)
test_fit(aug_h[12][aug_h[12].index.year.isin(range(1999,2010))].sknt)
#test_fit_weibull(aug_h[12][aug_h[12].index.year.isin(range(1999,2010)) & aug_h[12].drct.isin(range(45,135))].sknt)

def rotate_xaxis_labels(plot):

    x = plot.get_xaxis()
    for l in x.get_ticklabels():
        l.set_rotation(90.0)


# boxplot by hour
p = pd.DataFrame({h: aug_h[h][aug_h[h].index.year.isin(range(1999,2010))].sknt for h in range(24)}).boxplot()
p.set_title("Speed ranges by hour of day in August (1999-2009)")
p.set_ylim(-0.5,25.0)
show()

# THIS CANNOT BE RIGHT; IT HAS A SMALLER RANGE THAN THE FULL-YEAR DATA (see CY2000)!
# COULD WE BE LOOKING AT COUNTS NOT SPEEDS?

# boxplot august data by year
#pd.DataFrame({y: aug[aug.index.year == y].sknt for y in range(1999,2017)}).boxplot()
#figwidth = 10
figwidth = 20
p = pd.DataFrame({y: aug[aug.index.year == y].sknt for y in range(1993,2017)}).boxplot(figsize=(10,4))
title("Speed ranges by year, in August (1993-2016)")
p.set_ylim(-0.5,20.0)
rotate_xaxis_labels(p)
show()

# boxplot all data by year
#figsize = (10,4)
#figsize = (20,4)
figsize = (10,8)
p = pd.DataFrame({y: deduped_group[deduped_group.index.year == y].sknt for y in range(1993,2017)}).boxplot(figsize=(10,4))
p.set_title("Speed ranges by year (1993-2017)")
p.set_ylim(-0.5,20.0)
rotate_xaxis_labels(p)
show()

# Monthly graphs of average wind speed by hour

m_before = by_month(deduped_group['1994':'1998'])
m_after = by_month(deduped_group['2000':'2004'])
m_after2 = by_month(deduped_group['2005':'2009'])
m_after3 = by_month(deduped_group['2011':'2015'])


by_hour_each_month_before = pd.DataFrame(dict((mn, [hnd.sknt.mean() for (hn, hnd) in by_hour(m_before[mn]).items()]) for mn in m_before))
by_hour_each_month_before.columns = mpin[1:]
by_hour_each_month_before.columns = month_index
by_hour_each_month_after = pd.DataFrame(dict((mn, [hnd.sknt.mean() for (hn, hnd) in by_hour(m_after[mn]).items()]) for mn in m_before))
by_hour_each_month_after.columns = month_index

#by_hour_each_month_before["January"].plot()
#by_hour_each_month_after["January"].plot()
bva = pd.DataFrame({"Before": by_hour_each_month_before["January"],
                     "After": by_hour_each_month_after["January"]},
                      columns=["Before", "After"])
bva.plot()
plt.show()

# Average wind speed by month:

monthly_avgs_before = [m[mnum]['1994':'1998'].sknt.mean() for mnum in m]
monthly_avgs_after = [m[mnum]['2000':'2004'].sknt.mean() for mnum in m]
monthly_avgs_after2 = [m[mnum]['2005':'2009'].sknt.mean() for mnum in m]
monthly_avgs_after3 = [m[mnum]['2011':'2015'].sknt.mean() for mnum in m]

readings_during_burst_period = phx_df[(burst < phx_df.index) & (phx_df.index < reopened)]
m_dbp = by_month(readings_during_burst_period)
monthly_avgs_dbp = [m[mnum].sknt.mean() for mnum in m_dbp]
monthly_avgs_dbp_2 = [m[mnum].sknt.mean() if mnum in m_dbp else np.NaN for mnum in range(1,13)]
monthly_avgs_dbp = monthly_avgs_dbp_2

bva_df = pd.DataFrame({"Before": monthly_avgs_before, "After": monthly_avgs_after},
                      columns=["Before", "After"], index=mpin[1:13])
bva_df2 = pd.DataFrame({ "Before": monthly_avgs_before,
                         "After (2000-2004)": monthly_avgs_after,
                         "After (2005-2009)": monthly_avgs_after2,
                         "After (2011-2015)": monthly_avgs_after3,},
                       columns=[
                           "Before",
                           "After (2000-2004)",
                           "After (2005-2009)",
                           "After (2011-2015)"
                       ], index=mpin[1:13])

pbn = readings_during_burst_period.index.month.unique()
pbn.name = "Month"
if False:
    burst_df = pd.DataFrame({"While Burst (2010)": monthly_avgs_dbp},
                            columns=[
                             "While Burst (2010)",
                            ], index=pbn).reindex(mpins[1:13])
else:
    burst_df = pd.DataFrame({"While Burst (2010)": monthly_avgs_dbp},
                            columns=[
                             "While Burst (2010)",
                            ], index=mpin[1:13])
bva_df3 = pd.DataFrame({ "Before": monthly_avgs_before,
                         "After (2000-2004)": monthly_avgs_after,
                         "After (2005-2009)": monthly_avgs_after2,
                         "While Burst (2010)": monthly_avgs_dbp,
                         "After (2011-2015)": monthly_avgs_after3,},
                       columns=[
                           "Before",
                           "After (2000-2004)",
                           "After (2005-2009)",
                           "While Burst (2010)",
                           "After (2011-2015)"
                       ], index=mpin[1:13])

#bva_df.plot(use_index=True)
bva_df.plot(xlim=(-1,12), xticks=mpins)
plt.show()

bva_df2.plot(xlim=(-1,12), xticks=mpins)
plt.show()

bva_df3.plot(xlim=(-1,12), xticks=mpins)
plt.show()


# Exactly as in your example
bva_df.plot(xlim=(-1,12), xticks=mpins, style=["k", "k--"])
plt.show()

