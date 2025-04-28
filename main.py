# New forest

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error

# Import the data
model_table = pd.read_csv("data/forecast_data_merged.csv")

# Feature engineering functions
def add_cyclic_time_features(df, dt_col="valid_datetime"):
    df = df.copy()
    df[dt_col] = pd.to_datetime(df[dt_col])

    df["hour_sin"] = np.sin(2 * np.pi * df[dt_col].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df[dt_col].dt.hour / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df[dt_col].dt.weekday / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df[dt_col].dt.weekday / 7)
    df["month_sin"] = np.sin(2 * np.pi * df[dt_col].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df[dt_col].dt.month / 12)

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

# Pinball functions
def pinball(y,q,alpha):
    return (y-q)*alpha*(y>=q) + (q-y)*(1-alpha)*(y<q)

def pinball_score(df):
    score = list()
    for qu in range(10,100,10):
        score.append(pinball(y=df["total_generation_MWh"],
            q=df[f"q{qu}"],
            alpha=qu/100).mean())
    return sum(score)/len(score)

# Split the data into training and testing sets for solar and wind 
# Time gap that we're going to ignore
# Before: 2022-11-23 00:00:00
# After:  2022-12-06 18:00:00 

before_time = pd.Timestamp("2022-11-23 00:00:00+00:00")
after_time  = pd.Timestamp("2022-12-06 18:00:00+00:00")
end_time = pd.Timestamp("2023-10-27 23:30:00+00:00")

model_table['valid_datetime'] = pd.to_datetime(model_table['valid_datetime'])

train_and_valid = model_table[model_table["valid_datetime"] < before_time].copy()
test = model_table[(model_table["valid_datetime"] > after_time) & (model_table["valid_datetime"] <= end_time)].copy()

# Check the split
#print("Model table shape:", model_table.shape)  # (54348, 22)
#print("Train and valid shape:", train_and_valid.shape)  # (38112, 22)
#print("Test shape:", test.shape)    # (15575, 22)

# Split into solar and wind data
wind_columns = [
    "valid_datetime",
    "dwd_RelativeHumidity", "dwd_WindDirection_100", "dwd_WindSpeed_100",
    "ncep_RelativeHumidity", "ncep_WindDirection_100", "ncep_WindSpeed_100",
    "Wind_MW", "wind_curtailment_MW", "wind_potential_MW"
]

solar_columns = [
    "valid_datetime",
    "dwd_CloudCover", "dwd_SolarDownwardRadiation", "dwd_Temperature",
    "ncep_CloudCover", "ncep_SolarDownwardRadiation", "ncep_Temperature",
    "Solar_MW"
]
 
X_train_solar = train_and_valid[solar_columns].copy()
X_train_wind = train_and_valid[wind_columns].copy()
X_test_solar = test[solar_columns].copy()
X_test_wind = test[wind_columns].copy()

y_train_solar = train_and_valid["Solar_MWh_credit"].copy()
y_train_wind = train_and_valid["Wind_MWh_credit"].copy()
y_test_solar = test["Solar_MWh_credit"].copy()
y_test_wind = test["Wind_MWh_credit"].copy()

# Sort, is completely unnecessary but just in case
X_train_solar = X_train_solar.sort_values("valid_datetime")
X_train_wind = X_train_wind.sort_values("valid_datetime")
X_test_solar = X_test_solar.sort_values("valid_datetime")
X_test_wind = X_test_wind.sort_values("valid_datetime")

# add cyclic time features
X_train_solar = add_cyclic_time_features(X_train_solar, dt_col="valid_datetime")
X_train_wind = add_cyclic_time_features(X_train_wind, dt_col="valid_datetime")
X_test_solar = add_cyclic_time_features(X_test_solar, dt_col="valid_datetime")
X_test_wind = add_cyclic_time_features(X_test_wind, dt_col="valid_datetime")

# add wind direction cyclic features
for wind_col in ["dwd_WindDirection_100", "ncep_WindDirection_100"]:
    X_train_wind = add_wind_direction_cyclic(X_train_wind, wind_col)
    X_test_wind = add_wind_direction_cyclic(X_test_wind, wind_col)

# add lag features
sun_lag_columns = ["dwd_CloudCover", "dwd_SolarDownwardRadiation", "dwd_Temperature", 
                    "ncep_CloudCover", "ncep_SolarDownwardRadiation", "ncep_Temperature"]
wind_lag_columns = ["dwd_RelativeHumidity", "dwd_WindSpeed_100",
                     "ncep_RelativeHumidity","ncep_WindSpeed_100"]

X_train_solar = add_lag_features(X_train_solar, columns=sun_lag_columns, lags=[-2, -1, 0, 1, 2])
X_train_wind = add_lag_features(X_train_wind, columns=wind_lag_columns, lags=[-2, -1, 0, 1, 2])
X_test_solar = add_lag_features(X_test_solar, columns=sun_lag_columns, lags=[-2, -1, 0, 1, 2])
X_test_wind = add_lag_features(X_test_wind, columns=wind_lag_columns, lags=[-2, -1, 0, 1, 2])

# drop NaNs caused by lagging
X_train_solar = X_train_solar.dropna()
X_train_wind = X_train_wind.dropna()
X_test_solar = X_test_solar.dropna()
X_test_wind = X_test_wind.dropna()

# drop corresponding y rows to keep things aligned
y_train_solar = y_train_solar.loc[X_train_solar.index]
y_train_wind = y_train_wind.loc[X_train_wind.index]
y_test_solar = y_test_solar.loc[X_test_solar.index]
y_test_wind = y_test_wind.loc[X_test_wind.index]

# Remove the colums used for feautre engineering
X_train_solar = X_train_solar.drop(columns=["valid_datetime","dwd_CloudCover","dwd_SolarDownwardRadiation",
                                            "dwd_Temperature","ncep_CloudCover",
                                            "ncep_SolarDownwardRadiation","ncep_Temperature"])
X_train_wind = X_train_wind.drop(columns=["valid_datetime","dwd_RelativeHumidity","dwd_WindDirection_100",
                                            "dwd_WindSpeed_100","ncep_RelativeHumidity",
                                            "ncep_WindDirection_100","ncep_WindSpeed_100"])
X_test_solar = X_test_solar.drop(columns=["valid_datetime","dwd_CloudCover","dwd_SolarDownwardRadiation",
                                            "dwd_Temperature","ncep_CloudCover",
                                            "ncep_SolarDownwardRadiation","ncep_Temperature"])
X_test_wind = X_test_wind.drop(columns=["valid_datetime","dwd_RelativeHumidity","dwd_WindDirection_100",
                                            "dwd_WindSpeed_100","ncep_RelativeHumidity",
                                            "ncep_WindDirection_100","ncep_WindSpeed_100"])

# add scaleing 
X_train_solar, scaler = scale_features(X_train_solar)
X_test_solar, _ = scale_features(X_test_solar, scaler=scaler)
X_train_wind, scaler = scale_features(X_train_wind)
X_test_wind, _ = scale_features(X_test_wind, scaler=scaler)