# Welcome to compy and paste of prepross but with competition data

# Reqeuired Libraries
import pandas as pd
import xarray as xr
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# load data and create a dataframe
# DWD data
dwd_Hornsea = xr.open_dataset("data_comp/dwd_icon_eu_hornsea_1_20240129_20240519.nc")
subset = dwd_Hornsea[["RelativeHumidity", "WindDirection:100","WindSpeed:100"]]
avg_over_space = subset.mean(dim=["latitude","longitude"])
df_dwd_Hornsea = avg_over_space.to_dataframe().reset_index()
df_dwd_Hornsea["reference_time"] = df_dwd_Hornsea["reference_time"].dt.tz_localize("UTC")
df_dwd_Hornsea["valid_time"] = df_dwd_Hornsea["reference_time"] + pd.TimedeltaIndex(df_dwd_Hornsea["valid_time"],unit="hours")
df_dwd_Hornsea = df_dwd_Hornsea.rename(columns={
    "RelativeHumidity": "dwd_RelativeHumidity",
    "WindDirection:100": "dwd_WindDirection_100",
    "WindSpeed:100": "dwd_WindSpeed_100"
})

dwd_pes10 = xr.open_dataset("data_comp/dwd_icon_eu_pes10_20240129_20240519.nc")
subset = dwd_pes10[["CloudCover", "SolarDownwardRadiation","Temperature"]]
avg_over_space = subset.mean(dim=["point"])
df_dwd_pes10 = avg_over_space.to_dataframe().reset_index()
df_dwd_pes10["reference_time"] = df_dwd_pes10["reference_time"].dt.tz_localize("UTC")
df_dwd_pes10["valid_time"] = df_dwd_pes10["reference_time"] + pd.TimedeltaIndex(df_dwd_pes10["valid_time"],unit="hours")
df_dwd_pes10 = df_dwd_pes10.rename(columns={
    "CloudCover": "dwd_CloudCover",
    "SolarDownwardRadiation": "dwd_SolarDownwardRadiation",
    "Temperature": "dwd_Temperature"
})

# NCEP data
ncep_Hornsea = xr.open_dataset("data_comp/ncep_gfs_hornsea_1_20240129_20240519.nc")
subset = ncep_Hornsea[["RelativeHumidity", "WindDirection:100","WindSpeed:100"]]
avg_over_space = subset.mean(dim=["latitude","longitude"])
df_ncep_Hornsea = avg_over_space.to_dataframe().reset_index()
df_ncep_Hornsea["reference_time"] = df_ncep_Hornsea["reference_time"].dt.tz_localize("UTC")
df_ncep_Hornsea["valid_time"] = df_ncep_Hornsea["reference_time"] + pd.TimedeltaIndex(df_ncep_Hornsea["valid_time"],unit="hours")
df_ncep_Hornsea = df_ncep_Hornsea.rename(columns={
    "RelativeHumidity": "ncep_RelativeHumidity",
    "WindDirection:100": "ncep_WindDirection_100",
    "WindSpeed:100": "ncep_WindSpeed_100"
})

ncep_pes10 = xr.open_dataset("data_comp/ncep_gfs_pes10_20240129_20240519.nc")
subset = ncep_pes10[["CloudCover", "SolarDownwardRadiation","Temperature"]]
avg_over_space = subset.mean(dim=["point"])
df_ncep_pes10 = avg_over_space.to_dataframe().reset_index()
df_ncep_pes10["reference_time"] = df_ncep_pes10["reference_time"].dt.tz_localize("UTC")
df_ncep_pes10["valid_time"] = df_ncep_pes10["reference_time"] + pd.TimedeltaIndex(df_ncep_pes10["valid_time"],unit="hours")
df_ncep_pes10 = df_ncep_pes10.rename(columns={
    "CloudCover": "ncep_CloudCover",
    "SolarDownwardRadiation": "ncep_SolarDownwardRadiation",
    "Temperature": "ncep_Temperature"
})

# Merge dataframes
df_merged = df_dwd_Hornsea \
    .merge(df_dwd_pes10, how="outer",on=["reference_time","valid_time"]) \
    .merge(df_ncep_Hornsea, how="outer",on=["reference_time","valid_time"]) \
    .merge(df_ncep_pes10, how="outer",on=["reference_time","valid_time"])

df_merged = df_merged.set_index("valid_time").groupby("reference_time").resample("30T").interpolate("linear")
df_merged = df_merged.drop(columns="reference_time",axis=1).reset_index()

print("number of nans in the data:")
print(df_merged.isna().sum().sum())
# STILL No NaNs LETS GOO
# KEEP ALL OVERLAPP
#df_merged = df_merged[df_merged["valid_time"] - df_merged["reference_time"] < np.timedelta64(50,"h")]

# Save to CSVs
df_merged.to_csv("data_comp/forecast_data_merged_comp.csv", index=False)

