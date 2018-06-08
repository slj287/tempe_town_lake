#!/usr/bin/env python
"""simple wrapper for calling generate_deduped_winds to create ddg.csv files
"""

import sys

from wind_analysis import generate_deduped_winds


def generate_for_station(station_sym="PHX", drop_outliers=False):

    deduped_group = generate_deduped_winds(station_sym, drop_outliers)


if __name__ == "__main__":

    for station_key in sys.argv[1:]:
        generate_for_station(station_key, drop_outliers=False)

