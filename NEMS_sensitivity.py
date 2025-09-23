# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 14:44:48 2025

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
from RTM import NEMSv2

# This file contains the function used to create the sensitivity study

def bt_sensitivity_W(df):
    """
       This function computes the brightness temperature sensitivity to wind speed over open water.
    """
    # Constants
    theta = 0
    Ti_amsrv = np.full(7, 260)
    Ti_amsrh = np.full(7, 260)
    c_ice = 0
    e_icev = np.full(7, 0.9)
    e_iceh = np.full(7, 0.9)
    
    # Variables
    V = np.mean(df['V'])
    L = np.mean(df['L'])
    Ta = np.mean(df['Ta'])
    Ts = np.mean(df['Ts'])
    
    windspeeds = np.linspace(0, 20, 100)
    result_array = []
    channel1 = []
    
    for W in windspeeds:
        try:
            result = NEMSv2(V, W, L, Ta, Ts, theta, Ti_amsrv, Ti_amsrh, c_ice, e_icev, e_iceh)
            result_array.append(result)
        except Exception as e:
            print(f"Error processing data with wind speed {W}: {e}")
    
    for array in result_array:
        channel1.append(array[0])
    
    return windspeeds, channel1

def bt_sensitivity_V(df):
    """
    This function computes the brightness temperature sensitivity to Total Column Water Vapor (TCWV).
    """
    # Constants
    theta = 0
    Ti_amsrv = np.full(7, 260)
    Ti_amsrh = np.full(7, 260)
    c_ice = 0
    e_icev = np.full(7, 0.9)
    e_iceh = np.full(7, 0.9)
    
    # Variables
    L = np.mean(df['L'])
    Ta = np.mean(df['Ta'])
    Ts = np.mean(df['Ts'])
    W = np.mean(df['W'])
    
    V = np.linspace(0, 18, 100)
    result_array = []
    channel1 = []
    channel2 = []
    
    for value in V:
        try:
            result = NEMSv2(value, W, L, Ta, Ts, theta, Ti_amsrv, Ti_amsrh, c_ice, e_icev, e_iceh)
            result_array.append(result)
        except Exception as e:
            print(f"Error processing data with {value}: {e}")
    
    for array in result_array:
        channel1.append(array[0])
    
    return V, channel1

def bt_sensitivity_L(df):
    """
    This function computes the brightness temperature sensitivity to Total Column Water (TCW).
    """
    # Constants
    theta = 0
    Ti_amsrv = np.full(7, 260)
    Ti_amsrh = np.full(7, 260)
    c_ice = 0
    e_icev = np.full(7, 0.9)
    e_iceh = np.full(7, 0.9)
    
    # Variables
    V = np.mean(df['V'])
    Ta = np.mean(df['Ta'])
    Ts = np.mean(df['Ta'])
    W = np.mean(df['W']) 

    L = np.linspace(0, 0.3, 100)
    result_array = []
    channel1 = []
    channel2 = []
    
    for value in L:
        try:
            #result = scams(value, W, V, Ta, Ts, theta, Ti_amsrv, Ti_amsrh, c_ice, e_icev, e_iceh)
            result = NEMSv2(V, W, value, Ta, Ts, theta, Ti_amsrv, Ti_amsrh, c_ice, e_icev, e_iceh)
            result_array.append(result)
        except Exception as e:
            print(f"Error processing data with {value}: {e}")
    
    for array in result_array:
        channel1.append(array[0])
        
    return L, channel1

def bt_sensitivity_ts(df):
    """
    This function computes the brightness temperature sensitivity to Sea Surface Temperature (SST).
    """
    # Constants
    theta = 0
    Ti_amsrv = np.full(7, 260)
    Ti_amsrh = np.full(7, 260)
    c_ice = 0
    e_icev = np.full(7, 0.9)
    e_iceh = np.full(7, 0.9)
    
    # Variables
    V = np.mean(df['V'])
    Ta = np.mean(df['Ta'])
    L = np.mean(df['L'])
    W = np.mean(df['W'])
    
    Ts = np.linspace(270, 290, 100)
    result_array = []
    channel1 = []
    channel2 = []
    
    for value in Ts:
        try:
            #result = scams(value, W, V, Ta, L, theta, Ti_amsrv, Ti_amsrh, c_ice, e_icev, e_iceh)
            result = NEMSv2(V, W, L, Ta, value, theta, Ti_amsrv, Ti_amsrh, c_ice, e_icev, e_iceh)
            result_array.append(result)
        except Exception as e:
            print(f"Error processing data with {value}: {e}")
    
    for array in result_array:
        channel1.append(array[0])
    
    return Ts, channel1

def bt_sensitivity_ta(df):
    """
    This function computes the brightness temperature sensitivity to Sea Surface Temperature (SST).
    """
    # Constants
    theta = 0
    Ti_amsrv = np.full(7, 260)
    Ti_amsrh = np.full(7, 260)
    c_ice = 0
    e_icev = np.full(7, 0.9)
    e_iceh = np.full(7, 0.9)
    
    # Variables
    V = np.mean(df['V'])
    Ts = np.mean(df['Ta'])
    L = np.mean(df['L'])
    W = np.mean(df['W'])  
    
    Ta = np.linspace(230, 290, 100)
    result_array = []
    channel1 = []
    channel2 = []
    
    for value in Ta:
        try:
            #result = scams(value, W, V, Ts, L, theta, Ti_amsrv, Ti_amsrh, c_ice, e_icev, e_iceh)
            result = NEMSv2(V, W, L, value, Ts, Ti_amsrv, Ti_amsrh, c_ice, e_icev, e_iceh)
            result_array.append(result)
        except Exception as e:
            print(f"Error processing data with {value}: {e}")
    
    for array in result_array:
        channel1.append(array[0])
    
    return Ta, channel1

def plot_sensitivity_studies(df):
    """
    This function plots the sensitivity studies and boxplots on all 5 variables.
    """
    # Wind Speed Sensitivity
    windspeeds, wind_channel1, wind_channel2 = bt_sensitivity_W(df)
    plt.figure(figsize=(10, 8))
    plt.plot(windspeeds, wind_channel1, label='Channel 1', color='#8e9dd7', linewidth=4)
    plt.plot(windspeeds, wind_channel2, label='Channel 2', color='#e45a72', linewidth=4)
    plt.xlabel('[m/s]', fontsize=24)
    plt.ylabel('Brightness Temperature [K]', fontsize=24)
    plt.legend(loc='lower right', fontsize=20, frameon=True)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.show()

    # Wind boxplot
    plt.figure(figsize=(10, 3))
    plt.boxplot(df['W'], labels='W', patch_artist=True, boxprops=dict(facecolor='#8e9dd7'), medianprops=dict(color='black'),vert=False)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.xlabel('[m/s]', fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=18)
    
    # TCWV Sensitivity
    tcwv, tcwv_channel1, tcwv_channel2 = bt_sensitivity_L(df)
    plt.figure(figsize=(10, 8))
    plt.plot(tcwv, tcwv_channel1, label='Channel 1', color='#8e9dd7', linewidth=4)
    plt.plot(tcwv, tcwv_channel2, label='Channel 2', color='#e45a72', linewidth=4)
    plt.xlabel('[mm]', fontsize=24)
    plt.ylabel('Brightness Temperature [K]', fontsize=24)
    plt.legend(loc='lower right', fontsize=20, frameon=True)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.show()
    
    # L boxplot
    plt.figure(figsize=(10, 3))
    plt.boxplot(df['L'], labels='L', patch_artist=True, boxprops=dict(facecolor='#8e9dd7'), medianprops=dict(color='black'),vert=False)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.xlabel('[mm]', fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=18)

    # TCW Sensitivity
    tcw, tcw_channel1, tcw_channel2 = bt_sensitivity_V(df)
    plt.figure(figsize=(10, 8))
    plt.plot(tcw, tcw_channel1, label='Channel 1', color='#8e9dd7', linewidth=4)
    plt.plot(tcw, tcw_channel2, label='Channel 2', color='#e45a72', linewidth=4)
    plt.xlabel('[mm]', fontsize=24)
    plt.ylabel('Brightness Temperature [K]', fontsize=24)
    plt.legend(loc='lower right', fontsize=20, frameon=True)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.show()
    
    # V boxplot
    plt.figure(figsize=(10, 3))
    plt.boxplot(df['tcwv'], labels='V', patch_artist=True, boxprops=dict(facecolor='#8e9dd7'), medianprops=dict(color='black'),vert=False)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.xlabel('[mm]', fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=18)

    # SST Sensitivity
    sst, sst_channel1, sst_channel2 = bt_sensitivity_ts(df)
    plt.figure(figsize=(10, 8))
    plt.plot(sst, sst_channel1, label='Channel 1', color='#8e9dd7', linewidth=4)
    plt.plot(sst, sst_channel2, label='Channel 2', color='#e45a72', linewidth=4)
    plt.xlabel('[K]', fontsize=24)
    plt.ylabel('Brightness Temperature [K]', fontsize=24)
    plt.legend(loc='lower right', fontsize=20, frameon=True)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.show()
    
    # Ts boxplot
    plt.figure(figsize=(10, 3))
    plt.boxplot(df['sst'], labels=['Ts'], patch_artist=True, boxprops=dict(facecolor='#8e9dd7'), medianprops=dict(color='black'),vert=False)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.xlabel('[K]', fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=18)

    # T2M Sensitivity
    t2m, t2m_channel1, t2m_channel2 = bt_sensitivity_ta(df)
    plt.figure(figsize=(10, 8))
    plt.plot(t2m, t2m_channel1, label='Channel 1', color='#8e9dd7', linewidth=4)
    plt.plot(t2m, t2m_channel2, label='Channel 2', color='#e45a72', linewidth=4)
    plt.xlabel('[K]', fontsize=24)
    plt.ylabel('Brightness Temperature [K]', fontsize=24)
    plt.legend(loc='lower right', fontsize=20, frameon=True)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.show()  
    
    # Ta boxplot
    plt.figure(figsize=(10, 3))
    plt.boxplot(df['t2m'], labels=['Ta'], patch_artist=True, boxprops=dict(facecolor='#8e9dd7'), medianprops=dict(color='black'),vert=False)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.xlabel('[K]', fontsize=24)
    plt.tick_params(axis='both', which='major', labelsize=18)