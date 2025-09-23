import xarray as xr
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
from RTM import NEMSv3
from filters import filter_df
from ML import XGBoost_bias
import seaborn as sns

# Folder with NetCDFs and configuration
folder = r"C:\Users\user\OneDrive\Desktop\CCI\NEMS_colocated"
month = "m08"
channels = { 0: '22.2GHz',
             1: '31.4GHz'}
             # 2: '53.65GHz',
             # 3: '54.9GHz',
             # 4: '58.8GHz'}

# Find and sort files that match the month
files = sorted(glob.glob(os.path.join(folder, f"*{month}*.nc")))

# Open and combine them along the time dimension
ds = xr.open_mfdataset(files, combine="nested", concat_dim="t")

# Extract variables from the dataset
lat = ds["LAT"].isel(n16_frames=0).values
lon = ds["LON"].isel(n16_frames=0).values
time = ds["Time"].isel(n16_frames=0).values
V = ds["tcwv"].isel(n16_frames=0).values
Ta = ds["t2m"].isel(n16_frames=0).values
Ts = ds["sst"].isel(n16_frames=0).values
c_ice = ds["siconc"].isel(n16_frames=0).values
tcw = ds["tcw"].isel(n16_frames=0).values
L = tcw - V
W = np.sqrt(ds["v10"].isel(n16_frames=0).values**2 + ds["u10"].isel(n16_frames=0).values**2)

# Create a dictionary to hold the measured TB data for each channel
tb_measured_dict = {f"tb_measured_{i}": ds["TBNEMS"].isel(n14_frames=0, n5_channels=i).values for i in channels}

# Create dataframe
collected_df = pd.DataFrame({"V": V, "W": W, "L": L, "Ta": Ta, "Ts": Ts, "c_ice": c_ice, "lat": lat, "lon": lon})
collected_df = pd.concat([collected_df, pd.DataFrame(tb_measured_dict)], axis=1)

# Run biased NEMS RTM and assign to dataframe
tb_simulated = np.array([NEMSv3(V[i], W[i], L[i], Ta[i], Ts[i], c_ice[i]) for i in range(len(collected_df))])

for i in channels:
    collected_df[f'tb_simulated_{i}'] = tb_simulated[:, i]

# Filter data
clean_df = filter_df(collected_df.copy())

# Calculate bias for each channel
for i in channels:
    clean_df[f'bias_{i}'] = clean_df[f'tb_simulated_{i}'] - clean_df[f'tb_measured_{i}']

# Prepare features for the models
features = ['lat', 'lon', 'V', 'W', 'L', 'Ta', 'Ts', 'c_ice']

# Train the XGBoost models and store them in a dictionary
models = {}
for i in channels:
    print(f"Training model for channel {i}...")
    model = XGBoost_bias(df=clean_df,
            features=features,
            target=f'bias_{i}',
            test_size=0.2,
            random_state=42,
            n_estimators=550,
            max_depth=5,
            learning_rate=0.01,
            subsample=0.5,
            colsample_bytree=0.75)
    
    models[i] = model
    
    # Predict the bias and apply correction 
    predicted_bias = model.predict(clean_df[features])
    clean_df[f'tb_corrected_{i}'] = clean_df[f'tb_simulated_{i}'] - predicted_bias
    clean_df[f'tb_corrected_bias_{i}'] = clean_df[f'tb_corrected_{i}'] - clean_df[f'tb_measured_{i}']

print("Optimization complete. The corrected dataframe is ready.")


#%% Feature importance 
features = ['lat', 'lon', 'V', 'W', 'L', 'Ta', 'Ts', 'c_ice']

# Collect importance for all channels
importance_all = pd.DataFrame(index=features)

for i in channels:
    model = models[i]
    importance_dict = model.get_booster().get_score(importance_type='gain')
    importance_all[f'{channels[i]}'] = [importance_dict.get(f, 0) for f in features]

# Plot heatmap
plt.figure(figsize=(10,6))
sns.heatmap(importance_all, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Gain Importance Across Channels")
plt.ylabel("Feature")
plt.xlabel("Channel")
plt.show()

#%%
def bt_sensitivity(df, models, channels, param="W", apply_bias=True):
    """
    Compute brightness temperature sensitivity to a chosen parameter.
    Options: 'W', 'V', 'L', 'Ta', 'Ts'
    """
    # Range to era data
    # Compute average values from the dataframe
    V = 0
    L = 0
    Ta = np.mean(df['Ta'])
    Ts = np.mean(df['Ts'])
    W = np.mean(df['W'])

    # Define sweep ranges for each parameter
    param_ranges = {"W": np.linspace(df['W'].min(), df['W'].max(), 100),
                    "V": np.linspace(df['V'].min(), df['V'].max(), 100),
                    "L": np.linspace(df['L'].min(), df['L'].max(), 100),
                    "Ta": np.linspace(df['Ta'].min(), df['Ta'].max(), 100),
                    "Ts": np.linspace(df['Ts'].min(), df['Ts'].max(), 100)}
    
    sweep_values = param_ranges[param]

    # Store results per channel
    sensitivity_results = {i: [] for i in channels}

    # Compute average predicted bias per channel
    avg_bias = {}
    features = ['lat', 'lon', 'V', 'W', 'L', 'Ta', 'Ts', 'c_ice']
    if apply_bias:
        for i in channels:
            predicted_bias = models[i].predict(df[features])
            avg_bias[i] = np.mean(predicted_bias)

    for val in sweep_values:
        for i in channels:
            # Update the chosen parameter
            inputs = {
                "V": V,
                "W": W,
                "L": L,
                "Ta": Ta,
                "Ts": Ts,
                "c_ice": 0
            }
            inputs[param] = val

            # Simulate TB for this channel
            tb_sim = NEMSv3(**inputs)

            # Apply average bias correction if requested
            tb_corrected = tb_sim[i] - avg_bias[i] if apply_bias else tb_sim[i]
            sensitivity_results[i].append(tb_corrected)

    return sweep_values, sensitivity_results

sweep_values, sensitivity_results = bt_sensitivity(clean_df, models, channels, param="V")


plt.figure(figsize=(8, 5))
for i in channels:
    plt.plot(sweep_values, sensitivity_results[i], label=f"Channel {i}")
plt.xlabel("W")
plt.ylabel("Brightness Temperature (K)")
plt.title("Brightness Temperature Sensitivity to W")
plt.legend()
plt.grid(True)
plt.show()
