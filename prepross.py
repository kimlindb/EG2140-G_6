# Hello this file does five things
# 1. Imports raw data
# 2. Parses out the variables we will use
# 3. Fills in NaN values or time gaps
# 4. Fitts the data on the same time-axis (0.5h same as cvs file)
# 5. Splits the data into training, validation and testing

# More generly speaking this will take in the data and return the data will be used further down 

# 1. Parse Variables
# There are a total of 7 data sets, with a lot of variables and other jazz, 
# the implemented method will use ensables so most of the variables will be used, 
# things that will be dropped are related to traiding or redundancy

# import packages for testing file 
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# Loacal funcions
def interpolate_weather_variable(ds, var_name, point_idx, energy_data_dtm):
    """
    Interpolates a weather variable from an xarray.Dataset to match energy data timestamps.

    Parameters:
        ds (xarray.Dataset): The dataset containing weather data.
        var_name (str): Name of the variable in ds (e.g., "Temperature").
        point_idx (int): Index of the point to extract.
        energy_data_dtm (array-like): Timestamps of energy data.

    Returns:
        pd.Series: Interpolated weather variable values aligned with energy_data_dtm.
    """
    # Extract variable values and associated times
    var = ds[var_name][:, :, point_idx].values
    ref_times = pd.to_datetime(ds["ref_datetime"].values)
    valid_offsets = ds["valid_datetime"].values  # hours ahead

    # Flatten into timestamp-value pairs
    all_datetimes = []
    all_values = []
    for i, ref_time in enumerate(ref_times):
        for j, offset in enumerate(valid_offsets):
            forecast_time = ref_time + pd.Timedelta(hours=int(offset))
            value = var[i, j]
            if not np.isnan(value):
                all_datetimes.append(forecast_time)
                all_values.append(value)

    # Make sure energy timestamps are a proper DatetimeIndex
    energy_timestamps = pd.to_datetime(energy_data_dtm)
    energy_timestamps = pd.DatetimeIndex(energy_timestamps).tz_localize(None)

    # Create initial Series
    ts = pd.Series(all_values, index=pd.DatetimeIndex(all_datetimes)).tz_localize(None)

    # Average duplicates (if any)
    ts = ts.groupby(ts.index).mean()

    # Combine, sort and interpolate
    combined_index = pd.DatetimeIndex(ts.index.union(energy_timestamps)).sort_values()
    ts_interpolated = ts.reindex(combined_index).interpolate(method='time')

    # Return only the values aligned with energy timestamps
    return ts_interpolated.loc[energy_timestamps]


# 1. import data

# 1.1 Weather data 
dwd_demand = xr.open_dataset("data/dwd_icon_eu_demand_20200920_20231027.nc")
dwd_hornsea = xr.Dataset("data/dwd_icon_eu_hornsea_1_20200920_20231027.nc")
dwd_pes10 = xr.Dataset("data/dwd_icon_eu_pes10_20200920_20231027.nc")
ncep_demand = xr.Dataset("data/ncep_gfs_demand_20200920_20231027.nc")
ncep_hornsea = xr.Dataset("data/ncep_gfs_hornsea_1_20200920_20231027.nc")
ncep_pes10 = xr.Dataset("data/ncep_gfs_pes10_20200920_20231027.nc")

# 1.2 Energy data
energy_data = pd.read_csv("data/Energy_Data_20200920_20231027.csv") 

# 2.1 Parse energy data 
energy_data_dtm = energy_data["dtm"]
energy_data_Solar = energy_data["Solar_MW"]
energy_data_Wind = energy_data["Wind_MW"]

# 3.1 Repair energy data, there are no time gaps but some NaNs
# Simply use linear interpolisation 

energy_data_Solar = energy_data_Solar.sort_index()
energy_data_Solar.interpolate(method='linear', inplace=True)

energy_data_Wind = energy_data_Wind.sort_index()
energy_data_Wind.interpolate(method='linear', inplace=True)

# 3.2 Repair weather data, there are no time gaps but some NaNs

