
from datetime import datetime
import sys

import pandas as pd

from data_sources import in_out_file_map

indexable_timestamp = True

def generate_for_station(station_sym="PHX"):

    infile, index_col_for_this_file, outfile = in_out_file_map[station_sym]

    print("> Load raw readings from %s" % (infile))
    df = load_asos(infile, index_col=index_col_for_this_file)

    if indexable_timestamp:
        df["timestamp"] = pd.DatetimeIndex(df.index)
    else:
        df["timestamp"] = df.index

    print("> Narrowing to timestamp, drct, and sknt (+ dropping NaNs)")
    phx_df = df[df.sknt.notna() & df.drct.notna()][['timestamp', 'drct', 'sknt']]

    print("> Dedup readings")
    deduped_group = dedup_readings(phx_df)

    if "Hourly" in deduped_group.columns:
        deduped_group.index = deduped_group.Hourly

    cols_to_drop = (set(deduped_group.columns)
                    .intersection(set(["Hourly", "Hourly.1", "OffsetFromHour"])))
    if cols_to_drop:
        deduped_group.drop(cols_to_drop, axis=1, inplace=True)

    deduped_group.to_csv(outfile)
    print("> You now have a sparse set of hourly readings in %s." % (outfile))
    print("To get NaNs for hours without readings, phx_df.resample(\"H\")?")


if __name__ == "__main__":

    for station_key in sys.argv[1:]:
        generate_for_station(station_key)

