# DO NOT FORGET TO UNCOMMENT WRITE TO FILE
# FILE STORED IN THE DATA FOLDER

# Preporssesing of data is done like in competition
# Most of the code is copued from the competition notebook ("Getting Started.ipynb")
# and modified to fit the needs of this project

# Reqeuired Libraries
import pandas as pd
import xarray as xr
import numpy as np
#import statsmodels.formula.api as smf
#from statsmodels.iolib.smpickle import load_pickle
#import comp_utils
#import matplotlib.pyplot as plt
#import matplotlib.dates as mdates
#from matplotlib.ticker import MaxNLocator
#import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#import pickle as pkl


# load data and create a dataframe
# DWD data
dwd_Hornsea = xr.open_dataset("data/dwd_icon_eu_hornsea_1_20200920_20231027.nc")
subset = dwd_Hornsea[["RelativeHumidity", "WindDirection:100","WindSpeed:100"]]
avg_over_space = subset.mean(dim=["latitude","longitude"])
df_dwd_Hornsea = avg_over_space.to_dataframe().reset_index()
df_dwd_Hornsea["ref_datetime"] = df_dwd_Hornsea["ref_datetime"].dt.tz_localize("UTC")
df_dwd_Hornsea["valid_datetime"] = df_dwd_Hornsea["ref_datetime"] + pd.TimedeltaIndex(df_dwd_Hornsea["valid_datetime"],unit="hours")
df_dwd_Hornsea = df_dwd_Hornsea.rename(columns={
    "RelativeHumidity": "dwd_RelativeHumidity",
    "WindDirection:100": "dwd_WindDirection_100",
    "WindSpeed:100": "dwd_WindSpeed_100"
})

dwd_pes10 = xr.open_dataset("data/dwd_icon_eu_pes10_20200920_20231027.nc")
subset = dwd_pes10[["CloudCover", "SolarDownwardRadiation","Temperature"]]
avg_over_space = subset.mean(dim=["point"])
df_dwd_pes10 = avg_over_space.to_dataframe().reset_index()
df_dwd_pes10["ref_datetime"] = df_dwd_pes10["ref_datetime"].dt.tz_localize("UTC")
df_dwd_pes10["valid_datetime"] = df_dwd_pes10["ref_datetime"] + pd.TimedeltaIndex(df_dwd_pes10["valid_datetime"],unit="hours")
df_dwd_pes10 = df_dwd_pes10.rename(columns={
    "CloudCover": "dwd_CloudCover",
    "SolarDownwardRadiation": "dwd_SolarDownwardRadiation",
    "Temperature": "dwd_Temperature"
})

# NCEP data
ncep_Hornsea = xr.open_dataset("data/ncep_gfs_hornsea_1_20200920_20231027.nc")
subset = ncep_Hornsea[["RelativeHumidity", "WindDirection:100","WindSpeed:100"]]
avg_over_space = subset.mean(dim=["latitude","longitude"])
df_ncep_Hornsea = avg_over_space.to_dataframe().reset_index()
df_ncep_Hornsea["ref_datetime"] = df_ncep_Hornsea["ref_datetime"].dt.tz_localize("UTC")
df_ncep_Hornsea["valid_datetime"] = df_ncep_Hornsea["ref_datetime"] + pd.TimedeltaIndex(df_ncep_Hornsea["valid_datetime"],unit="hours")
df_ncep_Hornsea = df_ncep_Hornsea.rename(columns={
    "RelativeHumidity": "ncep_RelativeHumidity",
    "WindDirection:100": "ncep_WindDirection_100",
    "WindSpeed:100": "ncep_WindSpeed_100"
})

ncep_pes10 = xr.open_dataset("data/ncep_gfs_pes10_20200920_20231027.nc")
subset = ncep_pes10[["CloudCover", "SolarDownwardRadiation","Temperature"]]
avg_over_space = subset.mean(dim=["point"])
df_ncep_pes10 = avg_over_space.to_dataframe().reset_index()
df_ncep_pes10["ref_datetime"] = df_ncep_pes10["ref_datetime"].dt.tz_localize("UTC")
df_ncep_pes10["valid_datetime"] = df_ncep_pes10["ref_datetime"] + pd.TimedeltaIndex(df_ncep_pes10["valid_datetime"],unit="hours")
df_ncep_pes10 = df_ncep_pes10.rename(columns={
    "CloudCover": "ncep_CloudCover",
    "SolarDownwardRadiation": "ncep_SolarDownwardRadiation",
    "Temperature": "ncep_Temperature"
})

# Merge dataframes
df_merged = df_dwd_Hornsea \
    .merge(df_dwd_pes10, how="outer",on=["ref_datetime","valid_datetime"]) \
    .merge(df_ncep_Hornsea, how="outer",on=["ref_datetime","valid_datetime"]) \
    .merge(df_ncep_pes10, how="outer",on=["ref_datetime","valid_datetime"])

df_merged = df_merged.set_index("valid_datetime").groupby("ref_datetime").resample("30T").interpolate("linear")
df_merged = df_merged.drop(columns="ref_datetime",axis=1).reset_index()

# print("number of nans in the data:")
# print(df_merged.isna().sum().sum())
# No NaNs in the data great move on to energy data

# Load energy data
energy_data = pd.read_csv("data/Energy_Data_20200920_20231027.csv") 

energy_data_dtm = energy_data["dtm"]
energy_data_Solar = energy_data["Solar_MW"]
energy_data_Wind = energy_data["Wind_MW"]
energy_data_wind_curtailment = energy_data["boa_MWh"]

# Repair energy data, there are no time gaps but some NaNs, use linear interpolation
energy_data_Solar = energy_data_Solar.sort_index()
energy_data_Solar.interpolate(method='linear', inplace=True)
energy_data["Solar_MW"] = energy_data_Solar

energy_data_Wind = energy_data_Wind.sort_index()
energy_data_Wind.interpolate(method='linear', inplace=True)
energy_data["Wind_MW"] = energy_data_Wind

energy_data_wind_curtailment = energy_data_wind_curtailment.sort_index()
energy_data_wind_curtailment.interpolate(method='linear', inplace=True)
energy_data["boa_MWh"] = energy_data_wind_curtailment

# Calculate cutailment
energy_data['wind_curtailment_MW'] = -energy_data['boa_MWh'] / 0.5
energy_data['wind_potential_MW'] = energy_data['Wind_MW'] + energy_data['wind_curtailment_MW']

# TARGET MWh!!! 
energy_data["Wind_MWh_credit"] = 0.5*energy_data["Wind_MW"] - energy_data["boa_MWh"]
energy_data["Solar_MWh_credit"] = 0.5*energy_data["Solar_MW"]
energy_data["total_generation_MWh"] = energy_data["Wind_MWh_credit"] + energy_data["Solar_MWh_credit"]

energy_export = energy_data[[
    'dtm',
    'Solar_MW',
    'Wind_MW',
    'wind_curtailment_MW',
    'wind_potential_MW',
    'Solar_MWh_credit',
    'Wind_MWh_credit',
    'total_generation_MWh',
]]

# Convert dtm to datetime
energy_export = energy_export.copy()
energy_export['dtm'] = pd.to_datetime(energy_export['dtm'], utc=True)

df_merged = df_merged.merge(energy_export, how="inner", left_on="valid_datetime", right_on="dtm")
df_merged = df_merged[df_merged["valid_datetime"] - df_merged["ref_datetime"] < np.timedelta64(6,"h")]

# Save to CSVs
#df_merged.to_csv("data/forecast_data_merged.csv", index=False)


#print(len(energy_data_Wind)) # 54384
#print(len(df_merged)) # 54348
# They don't line up unforunately
# why?
# gud vet