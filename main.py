# hello world

# Time gap that we're going to ignore
# Before: 2022-11-23 00:00:00
# After:  2022-12-06 18:00:00

import time
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error

# import data

# weather data
weather_features = pd.read_csv("weather_features.csv")

# energy data
energy_data = pd.read_csv("data/Energy_Data_20200920_20231027.csv") 

def add_cyclic_time_features(df):
    """
    Adds cyclic time features (hour, month) using the DataFrame's datetime index.
    No need to drop any columns since the index is preserved.
    """
    # Ensure the DataFrame is a copy to avoid SettingWithCopyWarning
    df = df.copy()
    # Ensure the index is datetime (if not already)
    df.index = pd.to_datetime(df.index)
    
    # Extract time components from the index
    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df.index.weekday / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df.index.weekday / 7)
    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    
    return df

def add_wind_direction_cyclic(df, wind_dir_col="WindDirection"):
    df = df.copy()
    radians = np.deg2rad(df[wind_dir_col])
    df["WindDirection_sin"] = np.sin(radians)
    df["WindDirection_cos"] = np.cos(radians)
    return df

def add_lag_features(df, columns, lags=[1]):
    df = df.copy()
    for col in columns:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df

def scale_features(df, scaler=None):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if scaler is None:
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df, scaler

# Parse data
# Sun use pes10 dwd and ncep data, var CloudCover and SolarDownwardRadiation and Temperature
sun_weather_vars = ["dwd_pes10_CloudCover_P0","dwd_pes10_CloudCover_P1",
                    "dwd_pes10_CloudCover_P2","dwd_pes10_CloudCover_P3",
                    "dwd_pes10_CloudCover_P4","dwd_pes10_CloudCover_P5",
                    "dwd_pes10_CloudCover_P6","dwd_pes10_CloudCover_P7",
                    "dwd_pes10_CloudCover_P8","dwd_pes10_CloudCover_P9",
                    "dwd_pes10_CloudCover_P10","dwd_pes10_CloudCover_P11",
                    "dwd_pes10_CloudCover_P12","dwd_pes10_CloudCover_P13",
                    "dwd_pes10_CloudCover_P14","dwd_pes10_CloudCover_P15",
                    "dwd_pes10_CloudCover_P16","dwd_pes10_CloudCover_P17",
                    "dwd_pes10_CloudCover_P18","dwd_pes10_CloudCover_P19",
                    "dwd_pes10_SolarDownwardRadiation_P0",
                    "dwd_pes10_SolarDownwardRadiation_P1",
                    "dwd_pes10_SolarDownwardRadiation_P2",
                    "dwd_pes10_SolarDownwardRadiation_P3",
                    "dwd_pes10_SolarDownwardRadiation_P4",
                    "dwd_pes10_SolarDownwardRadiation_P5",
                    "dwd_pes10_SolarDownwardRadiation_P6",
                    "dwd_pes10_SolarDownwardRadiation_P7",
                    "dwd_pes10_SolarDownwardRadiation_P8",
                    "dwd_pes10_SolarDownwardRadiation_P9",
                    "dwd_pes10_SolarDownwardRadiation_P10",
                    "dwd_pes10_SolarDownwardRadiation_P11",
                    "dwd_pes10_SolarDownwardRadiation_P12",
                    "dwd_pes10_SolarDownwardRadiation_P13",
                    "dwd_pes10_SolarDownwardRadiation_P14",
                    "dwd_pes10_SolarDownwardRadiation_P15",
                    "dwd_pes10_SolarDownwardRadiation_P16",
                    "dwd_pes10_SolarDownwardRadiation_P17",
                    "dwd_pes10_SolarDownwardRadiation_P18",
                    "dwd_pes10_SolarDownwardRadiation_P19",
                    "dwd_pes10_Temperature_P0","dwd_pes10_Temperature_P1",
                    "dwd_pes10_Temperature_P2","dwd_pes10_Temperature_P3",
                    "dwd_pes10_Temperature_P4","dwd_pes10_Temperature_P5",
                    "dwd_pes10_Temperature_P6","dwd_pes10_Temperature_P7",
                    "dwd_pes10_Temperature_P8","dwd_pes10_Temperature_P9",
                    "dwd_pes10_Temperature_P10","dwd_pes10_Temperature_P11",
                    "dwd_pes10_Temperature_P12","dwd_pes10_Temperature_P13",
                    "dwd_pes10_Temperature_P14","dwd_pes10_Temperature_P15",
                    "dwd_pes10_Temperature_P16","dwd_pes10_Temperature_P17",
                    "dwd_pes10_Temperature_P18","dwd_pes10_Temperature_P19",
                    "ncep_pes10_CloudCover_P0","ncep_pes10_CloudCover_P1",
                    "ncep_pes10_CloudCover_P2","ncep_pes10_CloudCover_P3",
                    "ncep_pes10_CloudCover_P4","ncep_pes10_CloudCover_P5",
                    "ncep_pes10_CloudCover_P6","ncep_pes10_CloudCover_P7",
                    "ncep_pes10_CloudCover_P8","ncep_pes10_CloudCover_P9",
                    "ncep_pes10_CloudCover_P10","ncep_pes10_CloudCover_P11",
                    "ncep_pes10_CloudCover_P12","ncep_pes10_CloudCover_P13",
                    "ncep_pes10_CloudCover_P14","ncep_pes10_CloudCover_P15",
                    "ncep_pes10_CloudCover_P16","ncep_pes10_CloudCover_P17",
                    "ncep_pes10_CloudCover_P18","ncep_pes10_CloudCover_P19",
                    "ncep_pes10_SolarDownwardRadiation_P0",
                    "ncep_pes10_SolarDownwardRadiation_P1",
                    "ncep_pes10_SolarDownwardRadiation_P2",
                    "ncep_pes10_SolarDownwardRadiation_P3",
                    "ncep_pes10_SolarDownwardRadiation_P4",
                    "ncep_pes10_SolarDownwardRadiation_P5",
                    "ncep_pes10_SolarDownwardRadiation_P6",
                    "ncep_pes10_SolarDownwardRadiation_P7",
                    "ncep_pes10_SolarDownwardRadiation_P8",
                    "ncep_pes10_SolarDownwardRadiation_P9",
                    "ncep_pes10_SolarDownwardRadiation_P10",
                    "ncep_pes10_SolarDownwardRadiation_P11",
                    "ncep_pes10_SolarDownwardRadiation_P12",
                    "ncep_pes10_SolarDownwardRadiation_P13",
                    "ncep_pes10_SolarDownwardRadiation_P14",
                    "ncep_pes10_SolarDownwardRadiation_P15",
                    "ncep_pes10_SolarDownwardRadiation_P16",
                    "ncep_pes10_SolarDownwardRadiation_P17",
                    "ncep_pes10_SolarDownwardRadiation_P18",
                    "ncep_pes10_SolarDownwardRadiation_P19",
                    "ncep_pes10_Temperature_P0","ncep_pes10_Temperature_P1",
                    "ncep_pes10_Temperature_P2","ncep_pes10_Temperature_P3",
                    "ncep_pes10_Temperature_P4","ncep_pes10_Temperature_P5",
                    "ncep_pes10_Temperature_P6","ncep_pes10_Temperature_P7",
                    "ncep_pes10_Temperature_P8","ncep_pes10_Temperature_P9",
                    "ncep_pes10_Temperature_P10","ncep_pes10_Temperature_P11",
                    "ncep_pes10_Temperature_P12","ncep_pes10_Temperature_P13",
                    "ncep_pes10_Temperature_P14","ncep_pes10_Temperature_P15",
                    "ncep_pes10_Temperature_P16","ncep_pes10_Temperature_P17",
                    "ncep_pes10_Temperature_P18","ncep_pes10_Temperature_P19"
                    ]

# Wind use Hornsea dwd and ncep data, var RelativeHumidity and WindDirection and WindSpeed
wind_weather_vars = ["dwd_hornsea_RelativeHumidity_Lat0_Lon0",
                    "dwd_hornsea_RelativeHumidity_Lat0_Lon1",
                    "dwd_hornsea_RelativeHumidity_Lat0_Lon2",
                    "dwd_hornsea_RelativeHumidity_Lat0_Lon3",
                    "dwd_hornsea_RelativeHumidity_Lat0_Lon4",
                    "dwd_hornsea_RelativeHumidity_Lat0_Lon5",
                    "dwd_hornsea_RelativeHumidity_Lat1_Lon0",
                    "dwd_hornsea_RelativeHumidity_Lat1_Lon1",
                    "dwd_hornsea_RelativeHumidity_Lat1_Lon2",
                    "dwd_hornsea_RelativeHumidity_Lat1_Lon3",
                    "dwd_hornsea_RelativeHumidity_Lat1_Lon4",
                    "dwd_hornsea_RelativeHumidity_Lat1_Lon5",
                    "dwd_hornsea_RelativeHumidity_Lat2_Lon0",
                    "dwd_hornsea_RelativeHumidity_Lat2_Lon1",
                    "dwd_hornsea_RelativeHumidity_Lat2_Lon2",
                    "dwd_hornsea_RelativeHumidity_Lat2_Lon3",
                    "dwd_hornsea_RelativeHumidity_Lat2_Lon4",
                    "dwd_hornsea_RelativeHumidity_Lat2_Lon5",
                    "dwd_hornsea_RelativeHumidity_Lat3_Lon0",
                    "dwd_hornsea_RelativeHumidity_Lat3_Lon1",
                    "dwd_hornsea_RelativeHumidity_Lat3_Lon2",
                    "dwd_hornsea_RelativeHumidity_Lat3_Lon3",
                    "dwd_hornsea_RelativeHumidity_Lat3_Lon4",
                    "dwd_hornsea_RelativeHumidity_Lat3_Lon5",
                    "dwd_hornsea_RelativeHumidity_Lat4_Lon0",
                    "dwd_hornsea_RelativeHumidity_Lat4_Lon1",
                    "dwd_hornsea_RelativeHumidity_Lat4_Lon2",
                    "dwd_hornsea_RelativeHumidity_Lat4_Lon3",
                    "dwd_hornsea_RelativeHumidity_Lat4_Lon4",
                    "dwd_hornsea_RelativeHumidity_Lat4_Lon5",
                    "dwd_hornsea_RelativeHumidity_Lat5_Lon0",
                    "dwd_hornsea_RelativeHumidity_Lat5_Lon1",
                    "dwd_hornsea_RelativeHumidity_Lat5_Lon2",
                    "dwd_hornsea_RelativeHumidity_Lat5_Lon3",
                    "dwd_hornsea_RelativeHumidity_Lat5_Lon4",
                    "dwd_hornsea_RelativeHumidity_Lat5_Lon5",
                    "dwd_hornsea_WindDirection_Lat0_Lon0",
                    "dwd_hornsea_WindDirection_Lat0_Lon1",
                    "dwd_hornsea_WindDirection_Lat0_Lon2",
                    "dwd_hornsea_WindDirection_Lat0_Lon3",
                    "dwd_hornsea_WindDirection_Lat0_Lon4",
                    "dwd_hornsea_WindDirection_Lat0_Lon5",
                    "dwd_hornsea_WindDirection_Lat1_Lon0",
                    "dwd_hornsea_WindDirection_Lat1_Lon1",
                    "dwd_hornsea_WindDirection_Lat1_Lon2",
                    "dwd_hornsea_WindDirection_Lat1_Lon3",
                    "dwd_hornsea_WindDirection_Lat1_Lon4",
                    "dwd_hornsea_WindDirection_Lat1_Lon5",
                    "dwd_hornsea_WindDirection_Lat2_Lon0",
                    "dwd_hornsea_WindDirection_Lat2_Lon1",
                    "dwd_hornsea_WindDirection_Lat2_Lon2",
                    "dwd_hornsea_WindDirection_Lat2_Lon3",
                    "dwd_hornsea_WindDirection_Lat2_Lon4",
                    "dwd_hornsea_WindDirection_Lat2_Lon5",
                    "dwd_hornsea_WindDirection_Lat3_Lon0",
                    "dwd_hornsea_WindDirection_Lat3_Lon1",
                    "dwd_hornsea_WindDirection_Lat3_Lon2",
                    "dwd_hornsea_WindDirection_Lat3_Lon3",
                    "dwd_hornsea_WindDirection_Lat3_Lon4",
                    "dwd_hornsea_WindDirection_Lat3_Lon5",
                    "dwd_hornsea_WindDirection_Lat4_Lon0",
                    "dwd_hornsea_WindDirection_Lat4_Lon1",
                    "dwd_hornsea_WindDirection_Lat4_Lon2",
                    "dwd_hornsea_WindDirection_Lat4_Lon3",
                    "dwd_hornsea_WindDirection_Lat4_Lon4",
                    "dwd_hornsea_WindDirection_Lat4_Lon5",
                    "dwd_hornsea_WindDirection_Lat5_Lon0",
                    "dwd_hornsea_WindDirection_Lat5_Lon1",
                    "dwd_hornsea_WindDirection_Lat5_Lon2",
                    "dwd_hornsea_WindDirection_Lat5_Lon3",
                    "dwd_hornsea_WindDirection_Lat5_Lon4",
                    "dwd_hornsea_WindDirection_Lat5_Lon5",
                    "dwd_hornsea_WindSpeed_Lat0_Lon0",
                    "dwd_hornsea_WindSpeed_Lat0_Lon1",
                    "dwd_hornsea_WindSpeed_Lat0_Lon2",
                    "dwd_hornsea_WindSpeed_Lat0_Lon3",
                    "dwd_hornsea_WindSpeed_Lat0_Lon4",
                    "dwd_hornsea_WindSpeed_Lat0_Lon5",
                    "dwd_hornsea_WindSpeed_Lat1_Lon0",
                    "dwd_hornsea_WindSpeed_Lat1_Lon1",
                    "dwd_hornsea_WindSpeed_Lat1_Lon2",
                    "dwd_hornsea_WindSpeed_Lat1_Lon3",
                    "dwd_hornsea_WindSpeed_Lat1_Lon4",
                    "dwd_hornsea_WindSpeed_Lat1_Lon5",
                    "dwd_hornsea_WindSpeed_Lat2_Lon0",
                    "dwd_hornsea_WindSpeed_Lat2_Lon1",
                    "dwd_hornsea_WindSpeed_Lat2_Lon2",
                    "dwd_hornsea_WindSpeed_Lat2_Lon3",
                    "dwd_hornsea_WindSpeed_Lat2_Lon4",
                    "dwd_hornsea_WindSpeed_Lat2_Lon5",
                    "dwd_hornsea_WindSpeed_Lat3_Lon0",
                    "dwd_hornsea_WindSpeed_Lat3_Lon1",
                    "dwd_hornsea_WindSpeed_Lat3_Lon2",
                    "dwd_hornsea_WindSpeed_Lat3_Lon3",
                    "dwd_hornsea_WindSpeed_Lat3_Lon4",
                    "dwd_hornsea_WindSpeed_Lat3_Lon5",
                    "dwd_hornsea_WindSpeed_Lat4_Lon0",
                    "dwd_hornsea_WindSpeed_Lat4_Lon1",
                    "dwd_hornsea_WindSpeed_Lat4_Lon2",
                    "dwd_hornsea_WindSpeed_Lat4_Lon3",
                    "dwd_hornsea_WindSpeed_Lat4_Lon4",
                    "dwd_hornsea_WindSpeed_Lat4_Lon5",
                    "dwd_hornsea_WindSpeed_Lat5_Lon0",
                    "dwd_hornsea_WindSpeed_Lat5_Lon1",
                    "dwd_hornsea_WindSpeed_Lat5_Lon2",
                    "dwd_hornsea_WindSpeed_Lat5_Lon3",
                    "dwd_hornsea_WindSpeed_Lat5_Lon4",
                    "dwd_hornsea_WindSpeed_Lat5_Lon5",
                    "ncep_hornsea_RelativeHumidity_Lat0_Lon0",
                    "ncep_hornsea_RelativeHumidity_Lat0_Lon1",
                    "ncep_hornsea_RelativeHumidity_Lat0_Lon2",
                    "ncep_hornsea_RelativeHumidity_Lat1_Lon0",
                    "ncep_hornsea_RelativeHumidity_Lat1_Lon1",
                    "ncep_hornsea_RelativeHumidity_Lat1_Lon2",
                    "ncep_hornsea_RelativeHumidity_Lat2_Lon0",
                    "ncep_hornsea_RelativeHumidity_Lat2_Lon1",
                    "ncep_hornsea_RelativeHumidity_Lat2_Lon2",
                    "ncep_hornsea_WindDirection_Lat0_Lon0",
                    "ncep_hornsea_WindDirection_Lat0_Lon1",
                    "ncep_hornsea_WindDirection_Lat0_Lon2",
                    "ncep_hornsea_WindDirection_Lat1_Lon0",
                    "ncep_hornsea_WindDirection_Lat1_Lon1",
                    "ncep_hornsea_WindDirection_Lat1_Lon2",
                    "ncep_hornsea_WindDirection_Lat2_Lon0",
                    "ncep_hornsea_WindDirection_Lat2_Lon1",
                    "ncep_hornsea_WindDirection_Lat2_Lon2",
                    "ncep_hornsea_WindSpeed_Lat0_Lon0",
                    "ncep_hornsea_WindSpeed_Lat0_Lon1",
                    "ncep_hornsea_WindSpeed_Lat0_Lon2",
                    "ncep_hornsea_WindSpeed_Lat1_Lon0",
                    "ncep_hornsea_WindSpeed_Lat1_Lon1",
                    "ncep_hornsea_WindSpeed_Lat1_Lon2",
                    "ncep_hornsea_WindSpeed_Lat2_Lon0",
                    "ncep_hornsea_WindSpeed_Lat2_Lon1",
                    "ncep_hornsea_WindSpeed_Lat2_Lon2"
                    ]

# Energy
energy_data_dtm = energy_data["dtm"]
energy_data_Solar = energy_data["Solar_MW"]
energy_data_Wind = energy_data["Wind_MW"]

# Repair energy data, there are no time gaps but some NaNs
# Simply use linear interpolisation 

energy_data_Solar = energy_data_Solar.sort_index()
energy_data_Solar.interpolate(method='linear', inplace=True)

energy_data_Wind = energy_data_Wind.sort_index()
energy_data_Wind.interpolate(method='linear', inplace=True)

# Make sure the index is a DatetimeIndex
weather_features.index = pd.to_datetime(weather_features["dtm"])
weather_features = weather_features.drop(columns=["dtm"])

# Define split timestamps
before_time = pd.Timestamp("2022-11-23 00:00:00+00:00")
after_time  = pd.Timestamp("2022-12-06 18:00:00+00:00")

# Split the data
before_df = weather_features[weather_features.index < before_time]
after_df = weather_features[weather_features.index > after_time]

# Verify splits
#print("Before shape:", before_df.shape)  # Should exclude 2022-11-23 to 2022-12-06
#print("After shape:", after_df.shape)    # Should start after 2022-12-06
# Before shape: (38112, 370)
# After shape: (15611, 370)

# Select the relevant columns for solar and wind
# Solar data
weather_solar_before = before_df[sun_weather_vars]
weather_solar_after = after_df[sun_weather_vars]
# Wind data
weather_wind_before = before_df[wind_weather_vars]
weather_wind_after = after_df[wind_weather_vars]

# Ensure datetime parsing
energy_data_dtm = pd.to_datetime(energy_data["dtm"])

# Create the DataFrame with the datetime as the index
energy_df = pd.DataFrame({
    "Solar_MW": energy_data_Solar.values,
    "Wind_MW": energy_data_Wind.values
}, index=energy_data_dtm)

# Split the DataFrame
energy_before = energy_df[energy_df.index < before_time]
energy_after  = energy_df[energy_df.index > after_time]

# Create the training and testing sets

# Solar output
y_train_solar = energy_before["Solar_MW"].values
y_test_solar = energy_after["Solar_MW"].values

# Wind output
y_train_wind = energy_before["Wind_MW"].values
y_test_wind = energy_after["Wind_MW"].values

# Isolate single point data for training and testing
# For solar, use the first point (index 0) from the weather data


# Solar inputs 
X_train_solar_points = {}
X_scalars_solar_points = {}
X_test_solar_points = {}

# DWD data
for i in range(20):
    Xn_train_solar = weather_solar_before.iloc[:, [(0+i), (20+i), (40+i)]]
    Xn_train_solar = add_cyclic_time_features(Xn_train_solar)
    Xn_train_solar = add_lag_features(Xn_train_solar, [f"dwd_pes10_CloudCover_P{i}",
                                                    f"dwd_pes10_SolarDownwardRadiation_P{i}",
                                                    f"dwd_pes10_Temperature_P{i}"], lags=[-2, -1, 0, 1, 2])    
    Xn_train_solar, X_scalars_solar_points[i] = scale_features(Xn_train_solar)
    Xn_train_solar = Xn_train_solar.dropna()
    X_train_solar_points[i] = Xn_train_solar
    
    Xn_test_solar = weather_solar_after.iloc[:, [(0+i), (20+i), (40+i)]]
    Xn_test_solar = add_cyclic_time_features(Xn_test_solar)
    Xn_test_solar = add_lag_features(Xn_test_solar, [f"dwd_pes10_CloudCover_P{i}",
                                                    f"dwd_pes10_SolarDownwardRadiation_P{i}",
                                                    f"dwd_pes10_Temperature_P{i}"], lags=[-2, -1, 0, 1, 2])
    Xn_test_solar, _ = scale_features(Xn_test_solar)
    Xn_test_solar = Xn_test_solar.dropna()
    X_test_solar_points[i] = Xn_test_solar

# NCEP data
for i in range(20):
    Xn_train_solar = weather_solar_before.iloc[:, [(60+i), (80+i), (100+i)]]
    Xn_train_solar = add_cyclic_time_features(Xn_train_solar)
    Xn_train_solar = add_lag_features(Xn_train_solar, [f"ncep_pes10_CloudCover_P{i}",
                                                    f"ncep_pes10_SolarDownwardRadiation_P{i}",
                                                    f"ncep_pes10_Temperature_P{i}"], lags=[-2, -1, 0, 1, 2])    
    Xn_train_solar, X_scalars_solar_points[20+i] = scale_features(Xn_train_solar)
    Xn_train_solar = Xn_train_solar.dropna()
    X_train_solar_points[20+i] = Xn_train_solar
    
    Xn_test_solar = weather_solar_after.iloc[:, [(60+i), (80+i), (100+i)]]
    Xn_test_solar = add_cyclic_time_features(Xn_test_solar)
    Xn_test_solar = add_lag_features(Xn_test_solar, [f"ncep_pes10_CloudCover_P{i}",
                                                    f"ncep_pes10_SolarDownwardRadiation_P{i}",
                                                    f"ncep_pes10_Temperature_P{i}"], lags=[-2, -1, 0, 1, 2])
    Xn_test_solar, _ = scale_features(Xn_test_solar)
    Xn_test_solar = Xn_test_solar.dropna()
    X_test_solar_points[20+i] = Xn_test_solar

# Wind inputs 
X_train_wind_points = {}
X_scalars_wind_points = {}
X_test_wind_points = {}

# DWD data
idx = 0
for i in range(6):
    for j in range(6):
        Xn_train_wind = weather_wind_before.iloc[:, [(0+idx), (36+idx), (72+idx)]]
        Xn_train_wind = add_cyclic_time_features(Xn_train_wind)
        Xn_train_wind = add_wind_direction_cyclic(Xn_train_wind, wind_dir_col=f"dwd_hornsea_WindDirection_Lat{i}_Lon{j}")
        Xn_train_wind = add_lag_features(Xn_train_wind, [f"dwd_hornsea_RelativeHumidity_Lat{i}_Lon{j}",                
                                                        f"dwd_hornsea_WindSpeed_Lat{i}_Lon{j}"], lags=[-2, -1, 0, 1, 2])
        Xn_train_wind, X_scalars_wind_points[idx] = scale_features(Xn_train_wind)
        Xn_train_wind = Xn_train_wind.dropna()
        X_train_wind_points[idx] = Xn_train_wind

        Xn_test_wind = weather_wind_after.iloc[:, [(0+idx), (36+idx), (72+idx)]]
        Xn_test_wind = add_cyclic_time_features(Xn_test_wind)
        Xn_test_wind = add_wind_direction_cyclic(Xn_test_wind, wind_dir_col=f"dwd_hornsea_WindDirection_Lat{i}_Lon{j}")
        Xn_test_wind = add_lag_features(Xn_test_wind, [f"dwd_hornsea_RelativeHumidity_Lat{i}_Lon{j}",                
                                                        f"dwd_hornsea_WindSpeed_Lat{i}_Lon{j}"], lags=[-2, -1, 0, 1, 2])
        Xn_test_wind, _ = scale_features(Xn_test_wind)
        Xn_test_wind = Xn_test_wind.dropna()
        X_test_wind_points[idx] = Xn_test_wind
        idx += 1

# NCEP data
idx = 0
for i in range(3):
    for j in range(3):
        Xn_train_wind = weather_wind_before.iloc[:, [(108+idx), (117+idx), (126+idx)]]
        Xn_train_wind = add_cyclic_time_features(Xn_train_wind)
        Xn_train_wind = add_wind_direction_cyclic(Xn_train_wind, wind_dir_col=f"ncep_hornsea_WindDirection_Lat{i}_Lon{j}")
        Xn_train_wind = add_lag_features(Xn_train_wind, [f"ncep_hornsea_RelativeHumidity_Lat{i}_Lon{j}",                
                                                        f"ncep_hornsea_WindSpeed_Lat{i}_Lon{j}"], lags=[-2, -1, 0, 1, 2])
        Xn_train_wind, X_scalars_wind_points[36+idx] = scale_features(Xn_train_wind)
        Xn_train_wind = Xn_train_wind.dropna()
        X_train_wind_points[36+idx] = Xn_train_wind

        Xn_test_wind = weather_wind_after.iloc[:, [(108+idx), (117+idx), (126+idx)]]
        Xn_test_wind = add_cyclic_time_features(Xn_test_wind)
        Xn_test_wind = add_wind_direction_cyclic(Xn_test_wind, wind_dir_col=f"ncep_hornsea_WindDirection_Lat{i}_Lon{j}")
        Xn_test_wind = add_lag_features(Xn_test_wind, [f"ncep_hornsea_RelativeHumidity_Lat{i}_Lon{j}",                
                                                        f"ncep_hornsea_WindSpeed_Lat{i}_Lon{j}"], lags=[-2, -1, 0, 1, 2])
        Xn_test_wind, _ = scale_features(Xn_test_wind)
        Xn_test_wind = Xn_test_wind.dropna()
        X_test_wind_points[36+idx] = Xn_test_wind
        idx += 1

# Fun fact script unitl this point takes roughly 30 seconds to run


