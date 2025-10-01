import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from filters import filter_df
from data_visualization import dtu_blues_cmap, dtu_grey,dtu_coolwarm_cmap, dtu_red

file = r"C:\Users\user\OneDrive\Desktop\CCI\NEMS_colocated\Nimbus5-NEMS_L2_1973m1028t004449_DR27_era5.nc"

# Open and combine them along the time dimension
ds = xr.open_mfdataset(file)

#  Extract relevant variabels 
lat = ds["LAT"].isel(n16_frames=0).values
lon = ds["LON"].isel(n16_frames=0).values
tb_22 = ds["TBNEMS"].isel(n14_frames=0, n5_channels=0).values
tb_31 = ds["TBNEMS"].isel(n14_frames=0, n5_channels=1).values
c_ice = ds["siconc"].isel(n16_frames=0).values
sst = ds["sst"].isel(n16_frames=0).values

# Hemisphere masks
mask_north = lat >= 66.5
mask_south = lat <= -66.5

# Sort based on hemisphere
southeren_record =  pd.DataFrame({"tb_22": tb_22[mask_south], 
                                  "tb_31": tb_31[mask_south],
                                  "c_ice": c_ice[mask_south],
                                  "lat": lat[mask_south], 
                                  "lon": lon[mask_south],
                                  "sst":sst[mask_south]})

northeren_record =  pd.DataFrame({"tb_22": tb_22[mask_north], 
                                  "tb_31": tb_31[mask_north],
                                  "c_ice": c_ice[mask_north],
                                  "lat": lat[mask_north], 
                                  "lon": lon[mask_north],
                                  "sst":sst[mask_north]})

# Filter
southeren_record_filtered = filter_df(southeren_record)
northeren_record_filtered = filter_df(northeren_record)

# Classify based on GR
def classify_gr(gr):
    if gr > 0.015:
        return 'OW'
    elif -0.015 <= gr <= 0.015:
        return 'FYI'
    else:
        return 'MYI'

# Calculate GR
GR_southern = (southeren_record_filtered['tb_31'] - southeren_record_filtered['tb_22']) / (southeren_record_filtered['tb_31'] + southeren_record_filtered['tb_22'])
GR_northeren = (northeren_record_filtered['tb_31'] - northeren_record_filtered['tb_22']) / (northeren_record_filtered['tb_31'] + northeren_record_filtered['tb_22'])

# Add GR and categories to dataframes
southeren_record_filtered['GR'] = GR_southern
southeren_record_filtered['type'] = GR_southern.apply(classify_gr)

northeren_record_filtered['GR'] = GR_northeren
northeren_record_filtered['type'] = GR_northeren.apply(classify_gr)

#%% Calculate ice masks for high ice concentration (> 0.9)


southeren_record_filtered_ice = southeren_record_filtered[(southeren_record_filtered['c_ice'] >= 0.95)] 
northeren_record_filtered_ice = northeren_record_filtered[(northeren_record_filtered['c_ice'] >= 0.95)] 

print(southeren_record_filtered_ice)
#%%
# Tiepoints for FYI 
southern_fyi_tp22 = np.mean(southeren_record_filtered_ice.loc[southeren_record_filtered_ice['type'] == 'FYI', 'tb_22'])
southern_fyi_tp31 = np.mean(southeren_record_filtered_ice.loc[southeren_record_filtered_ice['type'] == 'FYI', 'tb_31'])
northern_fyi_tp22 = np.mean(northeren_record_filtered_ice.loc[northeren_record_filtered_ice['type'] == 'FYI', 'tb_22'])
northern_fyi_tp31 = np.mean(northeren_record_filtered_ice.loc[northeren_record_filtered_ice['type'] == 'FYI', 'tb_31'])

# Tiepoints for MYI
southern_myi_tp22 = np.mean(southeren_record_filtered_ice.loc[southeren_record_filtered_ice['type'] == 'MYI', 'tb_22'])
southern_myi_tp31 = np.mean(southeren_record_filtered_ice.loc[southeren_record_filtered_ice['type'] == 'MYI', 'tb_31'])
northern_myi_tp22 = np.mean(northeren_record_filtered_ice.loc[northeren_record_filtered_ice['type'] == 'MYI', 'tb_22'])
northern_myi_tp31 = np.mean(northeren_record_filtered_ice.loc[northeren_record_filtered_ice['type'] == 'MYI', 'tb_31'])

from filters import filter_by_area
import numpy as np

# Tiepoints for OW
southeren_record_filtered_sst = southeren_record_filtered[(southeren_record_filtered['sst'] <= 300)] 
northeren_record_filtered_sst = northeren_record_filtered[(northeren_record_filtered['sst'] <= 300)] 

northern_OW_tp22 = np.mean(southeren_record_filtered_sst.loc[southeren_record_filtered_sst['type'] == 'OW', 'tb_22'])
northern_OW_tp31 = np.mean(southeren_record_filtered_sst.loc[southeren_record_filtered_sst['type'] == 'OW', 'tb_31'])
southern_OW_tp22 = np.mean(southeren_record_filtered_sst.loc[southeren_record_filtered_sst['type'] == 'OW', 'tb_22'])
southern_OW_tp31 = np.mean(southeren_record_filtered_sst.loc[southeren_record_filtered_sst['type'] == 'OW', 'tb_31'])
print(southeren_record_filtered_sst)
# surface temp for vand - 278K (iskant) 

# Punkter pr dag
# Vand taet paa iskant med temperaturfilter
#  FYI
#%%
fig, axes = plt.subplots(1, 2, figsize=(10, 6),sharey=True)

print(southern_OW_tp31)
# Southern Hemisphere
ax = axes[0]
sc1 = ax.scatter(southeren_record_filtered['tb_31'], 
                 southeren_record_filtered['tb_22'], 
                 c=southeren_record_filtered['c_ice'], 
                 edgecolors="black",
                 cmap=dtu_blues_cmap, 
                 s=30)
ax.scatter(southern_myi_tp31, southern_myi_tp22, label='MYI', s=60, marker='x', color=dtu_red)
ax.scatter(southern_fyi_tp31, southern_fyi_tp22, label='FYI', s=60, marker='o', color=dtu_red)
ax.scatter(southern_OW_tp31, southern_OW_tp22, label='OW', s=60, marker='^', color=dtu_red)

ax.set_title('Southern Hemisphere')
ax.set_xlabel("tb_31")
ax.set_ylabel("tb_22")
ax.grid()
ax.legend()

# Northern Hemisphere
ax = axes[1]
sc2 = ax.scatter(northeren_record_filtered['tb_31'], 
                 northeren_record_filtered['tb_22'], 
                 c=northeren_record_filtered['c_ice'], 
                 edgecolors="black",
                 cmap=dtu_blues_cmap, 
                 s=30)
ax.scatter(northern_myi_tp31, northern_myi_tp22, label='MYI', s=60, marker='x', color=dtu_red)
ax.scatter(northern_fyi_tp31, northern_fyi_tp22, label='FYI', s=60, marker='o', color=dtu_red)
ax.scatter(northern_OW_tp31, northern_OW_tp22, label='OW', s=60, marker='^', color=dtu_red)

ax.set_title('Northern Hemisphere')
ax.set_xlabel("tb_31")
ax.grid()
ax.legend()

# Adjust layout so there's space at the bottom for colorbar
fig.subplots_adjust(bottom=0.2)

# Add a horizontal colorbar beneath both plots
cbar_ax = fig.add_axes([0.065, -0.05, 0.92, 0.05])  # [left, bottom, width, height]
cbar = fig.colorbar(sc2, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Ice Concentration [0-1]')

plt.tight_layout()
plt.show()



