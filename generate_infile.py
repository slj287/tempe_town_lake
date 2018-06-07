
from datetime import datetime
import sys

import pandas as pd

from data_sources import in_out_file_map
from wind_analysis import load_asos, narrow_asos_df_to_valid_hourly


def generate_for_station(station_sym="PHX"):

    infile, index_col_for_this_file, outfile = in_out_file_map[station_sym]

    print("> Load raw readings from %s" % (infile))
    df = load_asos(infile, index_col=index_col_for_this_file)


    deduped_group = narrow_asos_df_to_valid_hourly(df)

    deduped_group.to_csv(outfile)
    print("> You now have a sparse set of hourly readings in %s." % (outfile))
    print("To get NaNs for hours without readings, phx_df.resample(\"H\")?")


if __name__ == "__main__":

    for station_key in sys.argv[1:]:
        generate_for_station(station_key)

