import xarray as xr
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
from RTM import SCAMSv2
from data_visualization import plot_laea_antarctica
from filters import filter_df
import pandas as pd
from ML import XGBoost_bias


# Folder with NetCDFs
folder ="C:/Users/user/OneDrive/Desktop/CCI/SCAMS_colocated"
month = "m0301t002856"  # only load 1 file for now

# Find and sort files that match month
files = sorted(glob.glob(os.path.join(folder, f"*{month}*.nc")))

# Open and combine them along time dimension
ds = xr.open_mfdataset(files,
                       combine="nested",
                       concat_dim="Time")


# Access variables and check their shapes
# Desired datatype
dtype = np.float32

# Acces NEMS measurements
tb_measured = ds["TBCH1"].isel(n13_obs=0).values.astype(dtype)
lat = ds["LAT"].isel(n13_obs=0).values.astype(dtype)
lon = ds["LON"].isel(n13_obs=0).values.astype(dtype)
#time = ds["Time"].isel(n13_obs=0).values.astype(dtype)

# Extract variables from the dataset 
V = ds["tcwv"].isel(obs=0).values.astype(dtype) # in mm
Ta = ds["t2m"].isel(obs=0).values.astype(dtype)
Ts = ds["sst"].isel(obs=0).values.astype(dtype)
c_ice = ds["siconc"].isel(obs=0).values.astype(dtype)

# Calculate columnar cloud liquid water
tcw = ds["tcw"].isel(obs=0).values.astype(dtype)
L = tcw - V

# Calculate wind magnitude
v10 = ds["v10"].isel(obs=0).values.astype(dtype)
u10 = ds["u10"].isel(obs=0).values.astype(dtype)
W = np.sqrt(v10**2 + u10**2)

# Angle for feature
theta = 45

# Create dataframe for easy acces
collected_df = pd.DataFrame({"tb_measured":tb_measured,
                             "V": V,
                             "W": W,
                             "L": L,
                             "Ta": Ta,
                             "Ts": Ts,
                             "c_ice": c_ice,
                             "lat": lat,
                             "lon": lon})

# Run biased NEMS RTM
tb_simulated = []
tb_geometry = []

for i in range(len(tb_measured)):
    tb_NEMS = SCAMSv2(V[i], W[i], L[i], Ta[i], Ts[i], c_ice[i],theta)[0]  # 1 channel for now
    tb_simulated.append(tb_NEMS)
    
print(tb_geometry)

# Add simulated TB to dataframe
collected_df["tb_simulated"] = tb_simulated

# Filter measurements and climatology
clean_df = filter_df(collected_df)

# Calculate diffrence
clean_df['bias'] = clean_df['tb_simulated'] - clean_df['tb_measured']

# Train XGBoost model for bias correction of sensor
model = XGBoost_bias(df=clean_df, 
                     features=['lat', 'lon','V','W','L','Ta','Ts','c_ice'], 
                     target='bias')

# Predict the bias for all points using the trained XGBoost model
predicted_bias = model.predict(clean_df[['lat', 'lon','V','W','L','Ta','Ts','c_ice']])

# Apply the correction to the simulated TB
tb_corrected = clean_df['tb_simulated'] - predicted_bias

# Add corrected TB to the dataframe
clean_df['tb_corrected'] = tb_corrected

# Calculate the new bias in the corrected dataset
clean_df['tb_corrected_bias'] = clean_df['tb_corrected'] - clean_df['tb_measured']


#%%
print(clean_df['tb_corrected_bias'].describe())

#%%

# Plot histogram
plt.figure(figsize=(8,5))
plt.hist(clean_df['bias'], bins=50, color='skyblue', edgecolor='black')
plt.xlabel("TB simulated - TB measured")
plt.ylabel("Counts")
plt.title("Histogram of uncorrected bias")
plt.grid(True)
plt.show()

# Plot histogram
plt.figure(figsize=(8,5))
plt.hist(clean_df['tb_corrected_bias'], bins=50, color='skyblue', edgecolor='black')
plt.xlabel("TB simulated - TB measured")
plt.ylabel("Counts")
plt.title("Histogram of bias after correction")
plt.grid(True)
plt.show()



#%% Plotting!
plot_laea_antarctica(clean_df['lat'], clean_df['lon'], clean_df['tb_measured'], colorbar_min=160, colorbar_max=300)
plot_laea_antarctica(clean_df['lat'], clean_df['lon'], clean_df['tb_simulated'], colorbar_min=160, colorbar_max=300)
plot_laea_antarctica(clean_df['lat'], clean_df['lon'], clean_df['tb_corrected'], colorbar_min=160, colorbar_max=300)
