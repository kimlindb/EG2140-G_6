# Hello this file does five things
# 1. Imports raw data
# 2. Parses out the variables we will use
# 3. Fills in NaN values or time gaps
# 4. Fitts the data on the same time-axis (0.5h same as cvs file)
# 5. Splits the data into training, validation and testing

# More generly speaking this will take in the data and return the data will be used further down 

# 1. Parse Variables
# There are a total of 7 data sets, with a lot of variables and other jazz, the implemented method will use ensables so most of the variables will be used, things that will be dropped are related to traiding or redundancy


# import packages for testing file 
import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib.pyplot as plt

# 1. import data

# Weather data 
dwd_demand = nc.Dataset("data/dwd_icon_eu_demand_20200920_20231027.nc")
dwd_hornsea = nc.Dataset("data/dwd_icon_eu_hornsea_1_20200920_20231027.nc")
dwd_pes10 = nc.Dataset("data/dwd_icon_eu_pes10_20200920_20231027.nc")
ncep_demand = nc.Dataset("data/ncep_gfs_demand_20200920_20231027.nc")
ncep_hornsea = nc.Dataset("data/ncep_gfs_hornsea_1_20200920_20231027.nc")
ncep_pes10 = nc.Dataset("data/ncep_gfs_pes10_20200920_20231027.nc")

# energy data
energy_data = pd.read_csv("data/Energy_Data_20200920_20231027.csv") 

# 2. Parse variables
dwd_demand_RelativeHumidity = dwd_demand["RelativeHumidity"]
dwd_demand_Temperature = dwd_demand["Temperature"]
dwd_demand_TotalPrecipitation = dwd_demand["TotalPrecipitation"]
dwd_demand_WindDirection = dwd_demand["WindDirection"]
dwd_demand_WindSpeed = dwd_demand["WindSpeed"]

dwd_hornsea_RelativeHumidity = dwd_hornsea["RelativeHumidity"]
dwd_hornsea_Temperature = dwd_hornsea["Temperature"]
dwd_hornsea_WindDirection = dwd_hornsea["WindDirection"]
dwd_hornsea_WindDirection100 = dwd_hornsea["WindDirection:100"]
dwd_hornsea_WindSpeed = dwd_hornsea["WindSpeed"]
dwd_hornsea_WindSpeed100 = dwd_hornsea["WindSpeed:100"]

dwd_pes10_CloudCover = dwd_pes10["CloudCover"]
dwd_pes10_SolarDownwardRadiation = dwd_pes10["SolarDownwardRadiation"]
dwd_pes10_Temperature = dwd_pes10["Temperature"]

ncep_demand_RelativeHumidity = ncep_demand["RelativeHumidity"]
ncep_demand_Temperature = ncep_demand["Temperature"]
ncep_demand_TotalPrecipitation = ncep_demand["TotalPrecipitation"]
ncep_demand_WindDirection = ncep_demand["WindDirection"]
ncep_demand_WindSpeed = ncep_demand["WindSpeed"]

ncep_hornsea_RelativeHumidity = ncep_hornsea["RelativeHumidity"]
ncep_hornsea_Temperature = ncep_hornsea["Temperature"]
ncep_hornsea_WindDirection = ncep_hornsea["WindDirection"]
ncep_hornsea_WindDirection100 = ncep_hornsea["WindDirection:100"]
ncep_hornsea_WindSpeed = ncep_hornsea["WindSpeed"]
ncep_hornsea_WindSpeed100 = ncep_hornsea["WindSpeed:100"]

energy_data_dtm = energy_data["dtm"]
energy_data_Solar = energy_data["Solar_MW"]
energy_data_Wind = energy_data["Wind_MW"]

# 3.1 Repair energy data, there are no time gaps but some NaNs

# Simply use linear interpolisation 

energy_data_Solar = energy_data_Solar.sort_index()
energy_data_Solar.interpolate(method='linear', inplace=True)

energy_data_Wind = energy_data_Wind.sort_index()
energy_data_Wind.interpolate(method='linear', inplace=True)

# Now comes the shit of handeling the weather data


