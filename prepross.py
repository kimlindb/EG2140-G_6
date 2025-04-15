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
import xarray as xr
import matplotlib.pyplot as plt

# 1. import data

# Weather data 
dwd_demand = xr.open_dataset("data/dwd_icon_eu_demand_20200920_20231027.nc")
dwd_hornsea = xr.Dataset("data/dwd_icon_eu_hornsea_1_20200920_20231027.nc")
dwd_pes10 = xr.Dataset("data/dwd_icon_eu_pes10_20200920_20231027.nc")
ncep_demand = xr.Dataset("data/ncep_gfs_demand_20200920_20231027.nc")
ncep_hornsea = xr.Dataset("data/ncep_gfs_hornsea_1_20200920_20231027.nc")
ncep_pes10 = xr.Dataset("data/ncep_gfs_pes10_20200920_20231027.nc")



print("yes")
