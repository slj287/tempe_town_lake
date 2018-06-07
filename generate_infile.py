
from datetime import datetime
import sys

import pandas as pd

from data_sources import in_out_file_map
from wind_analysis import (load_asos, narrow_asos_df_to_valid_hourly,
                           find_extreme_outliers)


def generate_for_station(station_sym="PHX", drop_outliers=False):

    infile, index_col_for_this_file, outfile = in_out_file_map[station_sym]

    print("> Load raw readings from %s" % (infile))
    df = load_asos(infile, index_col=index_col_for_this_file)
    print(">   loaded %d reading(s)" % (len(df)))

    o = find_extreme_outliers(df)
    if o is not None:
        if drop_outliers:
            print("> Count of extreme outliers to drop: %d" % (len(o)))
            df = df.drop(o.index)
            print(">   count of reading(s) remaining: %d" % (len(df)))
        else:
            print("> Count of extreme outliers (not dropped): %d" % (len(o)))

    deduped_group = narrow_asos_df_to_valid_hourly(df)

    deduped_group.to_csv(outfile)
    print("> You now have a sparse set of hourly readings in %s." % (outfile))



if __name__ == "__main__":

    for station_key in sys.argv[1:]:
        generate_for_station(station_key, drop_outliers=False)

