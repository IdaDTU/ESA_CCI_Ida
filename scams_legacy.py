# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 10:35:54 2025

@author: user
"""

import numpy as np

import os
import fnmatch
import xarray as xr
import numpy as np
import sys, string
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os, string, math, cmath


def NEMSv2(V, W, L, Ta, Ts,c_ice):
    
    # NEMS freqensies
    frequencies = np.array([22.235, 31.400, 53.650, 54.900, 58.800])

    # Model Coefficients for the Atmosphere pr frequencies FOR AMSR!!!
    b0 = [241.69, 239.45, 245.87, 245.87, 245.87]
    b1 = [3.1032, 2.5441, 2.5061, 2.5061, 2.5061]
    b2 = [-0.081429, -0.051284, -0.062789, -0.062789, -0.062789]
    b3 = [0.00099893, 0.00045202, 0.00075962, 0.00075962, 0.00075962]
    b4 = [-4.837e-06, -1.436e-06, -3.606e-06, -3.606e-06, -3.606e-06]
    b5 = [0.20, 0.58, 0.53, 0.53, 0.53]
    b6 = [-0.20, -0.57, -12.52, -12.52, -12.52]
    b7 = [-0.0521, -0.0238, -0.2326, -0.2326, -0.2326]
    
    # RMS Error in Oxygen and Water Vapor Absorption Approximation FOR AMSR!!!
    ao1 = [0.01575, 0.04006, 1.13176, 1.13176, 1.13176]
    ao2 = [-0.000087, -0.000200, -0.000226, -0.000226, -0.000226]
    av1 = [0.00514, 0.00188, 0.00317, 0.00317, 0.00317]
    av2 = [0.19e-5, 0.09e-5, 0.27e-5, 0.27e-5, 0.27e-5]

    # Coefficients for Rayleigh Absorption and Mie Scattering FOR AMSR
    aL1 = [0.0891, 0.2027, 0.4021, 0.4021, 0.4021]
    aL2 = [0.0281, 0.0261, 0.0231, 0.0231, 0.0231]

    # Model Coefficients for Geometric Optics for AMSR
    r0v = [-0.00063, -0.00101, -0.00123, -0.00123, -0.00123]
    r0h = [0.00139, 0.00191, 0.00197, 0.00197, 0.00197]
    r1v = [-0.000070, -0.000105, -0.000113, -0.000113, -0.000113]
    r1h = [0.000085, 0.000112, 0.000119, 0.000119, 0.000119]
    r2v = [-2.1e-05, -2.1e-05, -2.1e-05, -2.1e-05, -2.1e-05]
    r2h = [-4.195e-05, -5.451e-05, -5.5e-05, -5.5e-05, -5.5e-05]
    r3v = [0.41e-06, 0.45e-06, 0.32e-06, 0.32e-06, 0.32e-06]
    r3h = [-0.20e-06, -0.36e-06, -0.44e-06, -0.44e-06, -0.44e-06]
    
        
    # The coefficients m1 and m2. Units are s/m. 
    m1v = [0.00178, 0.00257, 0.00260, 0.00260, 0.00260]
    m1h = [0.00308, 0.00329, 0.00330, 0.00330, 0.00330]
    m2v = [0.00730, 0.00701, 0.00700, 0.00700, 0.00700]
    m2h = [0.00660, 0.00660, 0.00660, 0.00660, 0.00660]
    
    # Ice temperatures and inv angle for freqs
    Ti_amsrv = np.full(5, 260)
    Ti_amsrh = np.full(5, 260)
    theta = 0
    e_icev = np.full(5, 0.9)
    e_iceh = np.full(5, 0.9)
    
    # --------------------------------------
    # Here the real simulation begins
    T_C = 2.7
    c_ice = max(0.0, min(c_ice, 1.0))
    Ts_mix = c_ice*Ta + (1-c_ice)*Ts
    Tl = (Ts_mix + 273.0)/2.0
    
    #eq 27 !!
    Tv_air = 273.16 + 0.8337*V - 3.029E-5*V**3.33
    if V > 48: Tv_air = 301.16
    G = 1.05*(Ts_mix-Tv_air)*(1-((Ts_mix-Tv_air)**2)/1200.0)
    if abs(Ts_mix-Tv_air) > 20: G = (Ts_mix-Tv_air)*14/abs(Ts_mix-Tv_air)

    epsilon_R = 4.44 # this value is from wentz and meisner, 2000, p. 28
    s = 35.0
    ny = 0.012  # Klein and Swift is using 0.02 which is giving a higher epsilon_R (4.9)
    light_speed = 3.0E10

    #eq 43
    epsilon_S = (87.90*np.exp(-0.004585*(Ts-273.15)))*(np.exp(-3.45E-3*s+4.69E-6*s**2+1.36E-5*s*(Ts-273.15)))
    
    #eq 44
    lambda_R = (3.30*np.exp(-0.0346*(Ts-273.15)+0.00017*(Ts-273.15)**2))-(6.54E-3*(1-3.06E-2*(Ts-273.15)+2.0E-4*(Ts-273.15)**2)*s)
    #eq 41
    C = 0.5536*s
    #eq 42
    delta_t = 25.0-(Ts-273.15)
    #eq 40
    qsi = 2.03E-2+1.27E-4*delta_t+2.46E-6*delta_t**2-C*(3.34E-5-4.60E-7*delta_t+4.60E-8*delta_t**2)
    #eq 39
    sigma = 3.39E9*(C**0.892)*np.exp(-delta_t*qsi)

    Tb = np.zeros(5)

    for i in range(5):
        #eq26
        # Downwelling radiation
        TD = b0[i]+b1[i]*V+b2[i]*V**2+b3[i]*V**3+b4[i]*V**4+b5[i]*G
        # Upwelling radiation
        TU = TD + b6[i]+b7[i]*V
        #eq 28
        AO = ao1[i]+ao2[i]*(TD-270.0)
        #eq 29
        AV = av1[i]*V + av2[i]*V**2
        #eq 33
        AL = aL1[i]*(1.0 - aL2[i]*(Tl-283.0))*L
        #eq 22
        tau = np.exp((-1.0/np.cos(np.radians(theta)))*(AO+AV+AL))
        #eq 24
        TBU = TU*(1.0-tau)
        TBD = TD*(1.0-tau)

        llambda = light_speed/(frequencies[i]*1e9)
        
        #eq 35
        epsilon = epsilon_R + ((epsilon_S-epsilon_R)/(1 + (1j*lambda_R/llambda)**(1-ny))) - (2*1j*sigma*llambda/light_speed)
        
        #eq.45
        sqrt_term = lib.scimath.sqrt(epsilon - np.sin(np.radians(theta))**2)
        rho_H = (np.cos(np.radians(theta)) - sqrt_term) / (np.cos(np.radians(theta)) + sqrt_term)
        rho_V = (epsilon*np.cos(np.radians(theta)) - sqrt_term) / (epsilon*np.cos(np.radians(theta)) + sqrt_term)
        
        #eq46
        R_0H = abs(rho_H)**2
        R_0V = abs(rho_V)**2 + (4.887E-8 - 6.108E-8*(Ts-273.0)**3)
        
        #eq 57
        R_geoH = R_0H - (r0h[i]+r1h[i]*(theta-53.0)+r2h[i]*(Ts-288.0)+r3h[i]*(theta-53.0)*(Ts-288.0))*W
        R_geoV = R_0V - (r0v[i]+r1v[i]*(theta-53.0)+r2v[i]*(Ts-288.0)+r3v[i]*(theta-53.0)*(Ts-288.0))*W
        
        # Check article!!
        #eq.60
        W1, W2 = 7.0, 12.0
        
        if W<W1: 
            F_H = m1h[i]*W
        elif W<W2:
            F_H = m1h[i]*W + 0.5*(m2h[i]-m1h[i])*((W-W1)**2)/(W2-W1)
        else: 
            F_H = m2h[i]*W - 0.5*(m2h[i]-m1h[i])*(W2+W1)

        W1, W2 = 3.0, 12.0
        
        if W<W1:
            F_V = m1v[i]*W
        elif W<W2:
            F_V = m1v[i]*W + 0.5*(m2v[i]-m1v[i])*((W-W1)**2)/(W2-W1)
        else:
            F_V = m2v[i]*W - 0.5*(m2v[i]-m1v[i])*(W2+W1)

        R_H = (1-F_H)*R_geoH
        R_V = (1-F_V)*R_geoV
        emissivityh = 1-R_H
        emissivityv = 1-R_V

        Delta_S2 = 5.22E-3*W if i>=4 else 5.22E-3*(1-0.00748*(37-frequencies[i])**1.3)*W
        Delta_S2 = min(Delta_S2,0.069)
        
        #eq.62
        term = Delta_S2 - 70.0*Delta_S2**3
        OmegaH = (6.2-0.001*(37-frequencies[i])**2)*term*tau**2
        OmegaV = (2.5+0.018*(37-frequencies[i]))*term*tau**3.4
        #eq.61
        T_BOmegaH = ((1+OmegaH)*(1-tau)*(TD-T_C)+T_C)*R_H
        T_BOmegaV = ((1+OmegaV)*(1-tau)*(TD-T_C)+T_C)*R_V

        Th = TBU + tau*((1-c_ice)*emissivityh*Ts + c_ice*e_iceh[i]*Ti_amsrh[i] + (1-c_ice)*(1-emissivityh)*(T_BOmegaH+tau*T_C) + c_ice*(1-e_iceh[i])*(TBD+tau*T_C))
        Tv = TBU + tau*((1-c_ice)*emissivityv*Ts + c_ice*e_icev[i]*Ti_amsrv[i] + (1-c_ice)*(1-emissivityv)*(T_BOmegaV+tau*T_C) + c_ice*(1-e_icev[i])*(TBD+tau*T_C))

        Tb[i] = Tv*(np.sin(np.radians(theta)))**2 + Th*(np.cos(np.radians(theta)))**2

    return Tb

def SCAMSv2(V, W, L, Ta, Ts,c_ice,theta):
    
    # NEMS freqensies
    frequencies = np.array([22.231, 31.650, 52.863, 53.845, 55.445])

    # Model Coefficients for the Atmosphere pr frequencies FOR AMSR!!!
    b0 = [241.69, 239.45, 245.87, 245.87, 245.87]
    b1 = [3.1032, 2.5441, 2.5061, 2.5061, 2.5061]
    b2 = [-0.081429, -0.051284, -0.062789, -0.062789, -0.062789]
    b3 = [0.00099893, 0.00045202, 0.00075962, 0.00075962, 0.00075962]
    b4 = [-4.837e-06, -1.436e-06, -3.606e-06, -3.606e-06, -3.606e-06]
    b5 = [0.20, 0.58, 0.53, 0.53, 0.53]
    b6 = [-0.20, -0.57, -12.52, -12.52, -12.52]
    b7 = [-0.0521, -0.0238, -0.2326, -0.2326, -0.2326]
    
    # RMS Error in Oxygen and Water Vapor Absorption Approximation FOR AMSR!!!
    ao1 = [0.01575, 0.04006, 1.13176, 1.13176, 1.13176]
    ao2 = [-0.000087, -0.000200, -0.000226, -0.000226, -0.000226]
    av1 = [0.00514, 0.00188, 0.00317, 0.00317, 0.00317]
    av2 = [0.19e-5, 0.09e-5, 0.27e-5, 0.27e-5, 0.27e-5]

    # Coefficients for Rayleigh Absorption and Mie ScatterinG FOR AMSR
    aL1 = [0.0891, 0.2027, 0.4021, 0.4021, 0.4021]
    aL2 = [0.0281, 0.0261, 0.0231, 0.0231, 0.0231]

    # Model Coefficients for Geometric Optics for AMSR
    r0v = [-0.00063, -0.00101, -0.00123, -0.00123, -0.00123]
    r0h = [0.00139, 0.00191, 0.00197, 0.00197, 0.00197]
    r1v = [-0.000070, -0.000105, -0.000113, -0.000113, -0.000113]
    r1h = [0.000085, 0.000112, 0.000119, 0.000119, 0.000119]
    r2v = [-2.1e-05, -2.1e-05, -2.1e-05, -2.1e-05, -2.1e-05]
    r2h = [-4.195e-05, -5.451e-05, -5.5e-05, -5.5e-05, -5.5e-05]
    r3v = [0.41e-06, 0.45e-06, 0.32e-06, 0.32e-06, 0.32e-06]
    r3h = [-0.20e-06, -0.36e-06, -0.44e-06, -0.44e-06, -0.44e-06]
    
        
    # The coefficients m1 and m2. Units are s/m. 
    m1v = [0.00178, 0.00257, 0.00260, 0.00260, 0.00260]
    m1h = [0.00308, 0.00329, 0.00330, 0.00330, 0.00330]
    m2v = [0.00730, 0.00701, 0.00700, 0.00700, 0.00700]
    m2h = [0.00660, 0.00660, 0.00660, 0.00660, 0.00660]
    
    # Ice temperatures for freqs
    Ti_amsrv = np.full(7, 260)
    Ti_amsrh = np.full(7, 260)
    e_icev = np.full(7, 0.9)
    e_iceh = np.full(7, 0.9)

    # --------------------------------------
    # Here the real simulation begins
    T_C = 2.7
    c_ice = max(0.0, min(c_ice, 1.0))
    Ts_mix = c_ice*Ta + (1-c_ice)*Ts
    Tl = (Ts_mix + 273.0)/2.0
    
    #eq 27 !!
    Tv_air = 273.16 + 0.8337*V - 3.029E-5*V**3.33
    if V > 48: Tv_air = 301.16
    G = 1.05*(Ts_mix-Tv_air)*(1-((Ts_mix-Tv_air)**2)/1200.0)
    if abs(Ts_mix-Tv_air) > 20: G = (Ts_mix-Tv_air)*14/abs(Ts_mix-Tv_air)

    epsilon_R = 4.44 # this value is from wentz and meisner, 2000, p. 28
    s = 35.0
    ny = 0.012  # Klein and Swift is using 0.02 which is giving a higher epsilon_R (4.9)
    light_speed = 3.0E10

    #eq 43
    epsilon_S = (87.90*np.exp(-0.004585*(Ts-273.15)))*(np.exp(-3.45E-3*s+4.69E-6*s**2+1.36E-5*s*(Ts-273.15)))
    
    #eq 44
    lambda_R = (3.30*np.exp(-0.0346*(Ts-273.15)+0.00017*(Ts-273.15)**2))-(6.54E-3*(1-3.06E-2*(Ts-273.15)+2.0E-4*(Ts-273.15)**2)*s)
    #eq 41
    C = 0.5536*s
    #eq 42
    delta_t = 25.0-(Ts-273.15)
    #eq 40
    qsi = 2.03E-2+1.27E-4*delta_t+2.46E-6*delta_t**2-C*(3.34E-5-4.60E-7*delta_t+4.60E-8*delta_t**2)
    #eq 39
    sigma = 3.39E9*(C**0.892)*np.exp(-delta_t*qsi)

    Tb = np.zeros(5)

    for i in range(5):
        #eq26
        # Downwelling radiation
        TD = b0[i]+b1[i]*V+b2[i]*V**2+b3[i]*V**3+b4[i]*V**4+b5[i]*G
        # Upwelling radiation
        TU = TD + b6[i]+b7[i]*V
        #eq 28
        AO = ao1[i]+ao2[i]*(TD-270.0)
        #eq 29
        AV = av1[i]*V + av2[i]*V**2
        #eq 33
        AL = aL1[i]*(1.0 - aL2[i]*(Tl-283.0))*L
        #eq 22
        tau = np.exp((-1.0/np.cos(np.radians(theta)))*(AO+AV+AL))
        #eq 24
        TBU = TU*(1.0-tau)
        TBD = TD*(1.0-tau)

        llambda = light_speed/(frequencies[i]*1e9)
        
        #eq 35
        epsilon = epsilon_R + ((epsilon_S-epsilon_R)/(1 + (1j*lambda_R/llambda)**(1-ny))) - (2*1j*sigma*llambda/light_speed)
        
        #eq.45
        rho_H = (np.cos(np.radians(theta)) - np.sqrt(epsilon - np.sin(np.radians(theta))**2)) / \
                (np.cos(np.radians(theta)) + np.sqrt(epsilon - np.sin(np.radians(theta))**2))
        rho_V = (epsilon*np.cos(np.radians(theta)) - np.sqrt(epsilon - np.sin(np.radians(theta))**2)) / \
                (epsilon*np.cos(np.radians(theta)) + np.sqrt(epsilon - np.sin(np.radians(theta))**2))
        
        #eq46
        R_0H = abs(rho_H)**2
        R_0V = abs(rho_V)**2 + (4.887E-8 - 6.108E-8*(Ts-273.0)**3)
        
        #eq 57
        R_geoH = R_0H - (r0h[i]+r1h[i]*(theta-53.0)+r2h[i]*(Ts-288.0)+r3h[i]*(theta-53.0)*(Ts-288.0))*W
        R_geoV = R_0V - (r0v[i]+r1v[i]*(theta-53.0)+r2v[i]*(Ts-288.0)+r3v[i]*(theta-53.0)*(Ts-288.0))*W
        
        # Check article!!
        #eq.60
        W1, W2 = 7.0, 12.0
        
        if W<W1: 
            F_H = m1h[i]*W
        elif W<W2:
            F_H = m1h[i]*W + 0.5*(m2h[i]-m1h[i])*((W-W1)**2)/(W2-W1)
        else: 
            F_H = m2h[i]*W - 0.5*(m2h[i]-m1h[i])*(W2+W1)

        W1, W2 = 3.0, 12.0
        
        if W<W1:
            F_V = m1v[i]*W
        elif W<W2:
            F_V = m1v[i]*W + 0.5*(m2v[i]-m1v[i])*((W-W1)**2)/(W2-W1)
        else:
            F_V = m2v[i]*W - 0.5*(m2v[i]-m1v[i])*(W2+W1)

        R_H = (1-F_H)*R_geoH
        R_V = (1-F_V)*R_geoV
        emissivityh = 1-R_H
        emissivityv = 1-R_V

        Delta_S2 = 5.22E-3*W if i>=4 else 5.22E-3*(1-0.00748*(37-frequencies[i])**1.3)*W
        Delta_S2 = min(Delta_S2,0.069)
        
        #eq.62
        term = Delta_S2 - 70.0*Delta_S2**3
        OmegaH = (6.2-0.001*(37-frequencies[i])**2)*term*tau**2
        OmegaV = (2.5+0.018*(37-frequencies[i]))*term*tau**3.4
        #eq.61
        T_BOmegaH = ((1+OmegaH)*(1-tau)*(TD-T_C)+T_C)*R_H
        T_BOmegaV = ((1+OmegaV)*(1-tau)*(TD-T_C)+T_C)*R_V

        Th = TBU + tau*((1-c_ice)*emissivityh*Ts + c_ice*e_iceh[i]*Ti_amsrh[i] + (1-c_ice)*(1-emissivityh)*(T_BOmegaH+tau*T_C) + c_ice*(1-e_iceh[i])*(TBD+tau*T_C))
        Tv = TBU + tau*((1-c_ice)*emissivityv*Ts + c_ice*e_icev[i]*Ti_amsrv[i] + (1-c_ice)*(1-emissivityv)*(T_BOmegaV+tau*T_C) + c_ice*(1-e_icev[i])*(TBD+tau*T_C))

        Tb[i] = Tv*(np.sin(np.radians(theta)))**2 + Th*(np.cos(np.radians(theta)))**2

    return Tb

def NEMS(V,W,L,Ta,Ts,c_ice):
    """
    Calculate the brightness temperatures at different microwave frequencies,
    considering the impact of atmospheric and surface variables on satellite microwave measurements,
    made for NEMS 5 dataset
    
    Args:
    V (float): Columnar water vapor [mm].(tcwv)
    W (float): Wind speed at the surface (m/s).
    L (float): Columnar cloud liquid water [mm] (tcw)
    Ta (float): Ambient air temperature (degrees Celsius).
    Ts (float): Sea surface temperature (degrees Celsius).
    Ti_amsr (array8): Ice Temperature [K] 
    c_ice (float): Ice concentration, ranging from 0 (no ice) to 1 (full ice cover).
    e_ice (array): Ice emmisivity

    Returns:
    Tb (array): Brightness temperatures for channels 22.231V, 31.650V, and 52.863V.
    """
    
    Ti_amsr,e_ice = np.full(7, 260), np.full(7, 0.9)

    # Frequency values for microwave measurements
    frequencies=np.array([6.93, 10.65, 18.70, 23.80, 36.50, 50.30, 52.80, 89.00])
    
    # Empirical coefficients for different calculations
    b0=np.array([239.50E+0,  239.51E+0,  240.24E+0,  241.69E+0,  239.45E+0,  242.10E+0,  245.87E+0,  242.58E+0])
    b1=np.array([213.92E-2,  225.19E-2,  298.88E-2,  310.32E-2,  254.41E-2,  229.17E-2,  250.61E-2,  302.33E-2])
    b2=np.array([-460.60E-4, -446.86E-4, -725.93E-4, -814.29E-4, -512.84E-4, -508.05E-4, -627.89E-4, -749.76E-4])
    b3=np.array([457.11E-6,  391.82E-6,  814.50E-6,  998.93E-6,  452.02E-6,  536.90E-6,  759.62E-6,  880.66E-6])
    b4=np.array([-16.84E-7,  -12.20E-7,  -36.07E-7,  -48.37E-7,  -14.36E-7,  -22.07E-7,  -36.06E-7,  -40.88E-7])
    b5=np.array([0.50E+0,     0.54E+0,    0.61E+0,    0.20E+0,    0.58E+0,    0.52E+0,    0.53E+0,    0.62E+0])
    b6=np.array([-0.11E+0,   -0.12E+0,   -0.16E+0,   -0.20E+0,   -0.57E+0,   -4.59E+0,  -12.52E+0,   -0.57E+0])
    b7=np.array([-0.21E-2,   -0.34E-2,   -1.69E-2,   -5.21E-2,   -2.38E-2,   -8.78E-2,  -23.26E-2,   -8.07E-2])
    ao1=np.array([8.34E-3,    9.08E-3,   12.15E-3,   15.75E-3,   40.06E-3,  353.72E-3, 1131.76E-3,   53.35E-3])
    ao2=np.array([-0.48E-4,  -0.47E-4,   -0.61E-4,   -0.87E-4,   -2.00E-4,  -13.79E-4,   -2.26E-4,   -1.18E-4])
    av1=np.array([0.07E-3,    0.18E-3,    1.73E-3,    5.14E-3,    1.88E-3,    2.91E-3,    3.17E-3,    8.78E-3])
    av2=np.array([0.00E-5,    0.00E-5,   -0.05E-5,    0.19E-5,    0.09E-5,    0.24E-5,    0.27E-5,    0.80E-5])

    aL1=np.array([0.0078, 0.0183, 0.0556, 0.0891,  0.2027,  0.3682,  0.4021,  0.9693])
    aL2=np.array([0.0303, 0.0298, 0.0288, 0.0281,  0.0261,  0.0236,  0.0231,  0.0146])
    aL3=np.array([0.0007, 0.0027, 0.0113, 0.0188,  0.0425,  0.0731,  0.0786,  0.1506])
    aL4=np.array([0.0000, 0.0060, 0.0040, 0.0020, -0.0020, -0.0020, -0.0020, -0.0020])
    aL5=np.array([1.2216, 1.1795, 1.0636, 1.0220,  0.9546,  0.8983,  0.8943,  0.7961])

    r0v=np.array([-0.27E-3,  -0.32E-3,  -0.49E-3,  -0.63E-3,  -1.01E-3, -1.20E-3, -1.23E-03, -1.53E-3])
    r0h=np.array([0.54E-3,   0.72E-3,   1.13E-3,   1.39E-3,   1.91E-3,  1.97E-3,  1.97E-03,  2.02E-3])
    r1v=np.array([-0.21E-4,  -0.29E-4,  -0.53E-4,  -0.70E-4,  -1.05E-4, -1.12E-4, -1.13E-04, -1.16E-4])
    r1h=np.array([0.32E-4,   0.44E-4,   0.70E-4,   0.85E-4,   1.12E-4,  1.18E-4,  1.19E-04,  1.30E-4])
    r2v=np.array([-2.10E-5,  -2.10E-5,  -2.10E-5,  -2.10E-5,  -2.10E-5, -2.10E-5, -2.10E-05, -2.10E-5])
    r2h=np.array([-25.26E-6, -28.94E-6, -36.90E-6, -41.95E-6, -54.51E-6, -5.50E-5, -5.50E-5,  -5.50E-5])
    r3v=np.array([0.00E-6,   0.08E-6,   0.31E-6,   0.41E-6,   0.45E-6,  0.35E-6,  0.32E-06, -0.09E-6])
    r3h=np.array([0.00E-6,  -0.02E-6,  -0.12E-6,  -0.20E-6,  -0.36E-6, -0.43E-6, -0.44E-06, -0.46E-6])

    m1v=np.array([0.00020, 0.00020, 0.00140, 0.00178, 0.00257, 0.00260, 0.00260, 0.00260])
    m1h=np.array([0.00200, 0.00200, 0.00293, 0.00308, 0.00329, 0.00330, 0.00330, 0.00330])
    m2v=np.array([0.00690, 0.00690, 0.00736, 0.00730, 0.00701, 0.00700, 0.00700, 0.00700])
    m2h=np.array([0.00600, 0.00600, 0.00656, 0.00660, 0.00660, 0.00660, 0.00660, 0.00660])
    
    # Arrays for storing calculated values
    TD=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    TU=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    AO=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    AV=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    AL=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    tau=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    TBU=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    TBD=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    llambda=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    epsilon=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], dtype=np.complex128)
    rho_H=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], dtype=np.complex128)
    rho_V=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], dtype=np.complex128)
    R_0H=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    R_0V=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    R_geoH=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    R_geoV=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    F_H=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    F_V=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    R_H=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    R_V=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    OmegaH=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    OmegaV=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    T_BOmegaH=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    T_BOmegaV=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    emissivityh=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    emissivityv=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    term=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    Delta_S2=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    Tv=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    Th=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

    T_C=2.7

    if c_ice < 0.0: c_ice=0.0
    if c_ice > 1.0: c_ice=1.0

    Ts_mix=c_ice*Ta+(1.0-c_ice)*Ts

    Tl=(Ts_mix+273.0)/2.0

    #eq 27
    Tv=273.16+0.8337*V-3.029E-5*(V**3.33)
    if V > 48:  Tv=301.16
    G = 1.05*(Ts_mix-Tv)*(1-((Ts_mix-Tv)**2)/1200.0)
    if math.fabs(Ts_mix-Tv) > 20: G = (Ts_mix-Tv)*14/math.fabs(Ts_mix-Tv)

    epsilon_R=4.44 # this value is from wentz and meisner, 2000, p. 28
    s=35.0
    ny=0.012 # Klein and Swift is using 0.02 which is giving a higher epsilon_R (4.9)
    light_speed=3.00E10
    free_space_permittivity=8.854E-12
    #eq 43
    epsilon_S = (87.90*math.exp(-0.004585*(Ts-273.15)))*(math.exp(-3.45E-3*s+4.69E-6*s**2+1.36E-5*s*(Ts-273.15)))
    #eq 44
    lambda_R = (3.30*math.exp(-0.0346*(Ts-273.15)+0.00017*(Ts-273.15)**2))-(6.54E-3*(1-3.06E-2*(Ts-273.15)+2.0E-4*(Ts-273.15)**2)*s)
    #eq 41
    C=0.5536*s
    #eq 42
    delta_t=25.0-(Ts-273.15)
    #eq 40
    qsi=2.03E-2+1.27E-4*delta_t+2.46E-6*delta_t**2-C*(3.34E-5-4.60E-7*delta_t+4.60E-8*delta_t**2)
    #eq 39
    sigma=3.39E9*(C**0.892)*math.exp(-delta_t*qsi)

    for i in range(0,8):
        #eq26
        TD[i]=b0[i]+b1[i]*V+b2[i]*V**2+b3[i]*V**3+b4[i]*V**4+b5[i]*G
        TU[i]=TD[i]+b6[i]+b7[i]*V
        #eq 28
        AO[i]=ao1[i]+ao2[i]*(TD[i]-270.0)
        #eq 29
        AV[i]=av1[i]*V+av2[i]*V**2
        #eq 33
        AL[i]=aL1[i]*(1.0-aL2[i]*(Tl-283.0))*L
        #eq 22
        tau[i]=np.exp((-1.0/math.cos(math.radians(0)))*(AO[i]+AV[i]+AL[i])) 
        #eq 24
        TBU[i]=TU[i]*(1.0-tau[i])
        TBD[i]=TD[i]*(1.0-tau[i])

        llambda[i]=(light_speed/(frequencies[i]*1E9))

        #eq 35
        epsilon[i]=epsilon_R+((epsilon_S-epsilon_R)/(1.0+((cmath.sqrt(-1)*lambda_R)/llambda[i])**(1.0-ny)))-((2.0*cmath.sqrt(-1)*sigma*llambda[i])/light_speed)
        #eq.45
        rho_H[i]=(math.cos(math.radians(0))-cmath.sqrt(epsilon[i]-math.sin(math.radians(0))**2))/\
                 (math.cos(math.radians(0))+cmath.sqrt(epsilon[i]-math.sin(math.radians(0))**2))
        rho_V[i]=(epsilon[i]*math.cos(math.radians(0))-cmath.sqrt(epsilon[i]-math.sin(math.radians(0))**2))/\
                 (epsilon[i]*math.cos(math.radians(0))+cmath.sqrt(epsilon[i]-math.sin(math.radians(0))**2))
        #eq46
        R_0H[i]=np.absolute(rho_H[i])**2
        R_0V[i]=np.absolute(rho_V[i])**2+(4.887E-8-6.108E-8*(Ts-273.0)**3)

        #eq 57
        R_geoH[i]=R_0H[i]-(r0h[i]+r1h[i]*(0-53.0)+r2h[i]*(Ts-288.0)+r3h[i]*(0-53.0)*(Ts-288.0))*W
        R_geoV[i]=R_0V[i]-(r0v[i]+r1v[i]*(0-53.0)+r2v[i]*(Ts-288.0)+r3v[i]*(0-53.0)*(Ts-288.0))*W
        #eq.60
        W_1=7.0
        W_2=12.0
        if W<W_1: F_H[i]=m1h[i]*W
        elif W_1<W<W_2: F_H[i]=m1h[i]*W+0.5*(m2h[i]-m1h[i])*((W-W_1)**2)/(W_2-W_1)
        else: F_H[i]=m2h[i]*W-0.5*(m2h[i]-m1h[i])*(W_2+W_1)
        W_1=3.0
        W_2=12.0
        if W<W_1: F_V[i]=m1v[i]*W
        elif W_1<W<W_2: F_V[i]=m1v[i]*W+0.5*(m2v[i]-m1v[i])*((W-W_1)**2)/(W_2-W_1)
        else: F_V[i]=m2v[i]*W-0.5*(m2v[i]-m1v[i])*(W_2+W_1)

        R_H[i]=(1-F_H[i])*R_geoH[i]
        R_V[i]=(1-F_V[i])*R_geoV[i]

        emissivityh[i]=1-R_H[i]
        emissivityv[i]=1-R_V[i]

        if i >= 4: Delta_S2[i]=5.22E-3*W
        else: Delta_S2[i]=5.22E-3*(1-0.00748*(37.0-frequencies[i])**1.3)*W
        if Delta_S2[i]>0.069: Delta_S2[i]=0.069
        #eq.62
        term[i]=Delta_S2[i]-70.0*Delta_S2[i]**3
        OmegaH[i]=(6.2-0.001*(37.0-frequencies[i])**2)*term[i]*tau[i]**2.0
        OmegaV[i]=(2.5+0.018*(37.0-frequencies[i]))*term[i]*tau[i]**3.4
        #eq.61
        T_BOmegaH[i]=((1+OmegaH[i])*(1-tau[i])*(TD[i]-T_C)+T_C)*R_H[i] 
        T_BOmegaV[i]=((1+OmegaV[i])*(1-tau[i])*(TD[i]-T_C)+T_C)*R_V[i]
        
        
        Th22=TBU[0]+tau[0]*((1.0 - c_ice)*emissivityh[0]*Ts+c_ice*e_ice[0]*Ti_amsr[0]+(1.0 - c_ice)*(1.0 - emissivityh[0])*\
             (T_BOmegaH[0]+tau[0]*T_C)+c_ice*(1.0 - e_ice[0])*(TBD[0]+tau[0]*T_C))

        Tv22=TBU[0]+tau[0]*((1.0 - c_ice)*emissivityv[0]*Ts+c_ice*e_ice[0]*Ti_amsr[0]+(1.0 - c_ice)*(1.0 - emissivityv[0])*\
             (T_BOmegaV[0]+tau[0]*T_C)+c_ice*(1.0 - e_ice[0])*(TBD[0]+tau[0]*T_C))
        
        T22=Tv22*(math.sin(math.radians(0)))**2 + Th22*(math.cos(math.radians(0)))**2
        
        Th31=TBU[4]+tau[4]*((1.0 - c_ice)*emissivityh[4]*Ts+c_ice*e_ice[4]*Ti_amsr[4]+(1.0 - c_ice)*(1.0 - emissivityh[4])*\
             (T_BOmegaH[4]+tau[4]*T_C)+c_ice*(1.0 - e_ice[4])*(TBD[4]+tau[4]*T_C))

        Tv31=TBU[4]+tau[4]*((1.0 - c_ice)*emissivityv[4]*Ts+c_ice*e_ice[4]*Ti_amsr[4]+(1.0 - c_ice)*(1.0 - emissivityv[4])*\
             (T_BOmegaV[4]+tau[4]*T_C)+c_ice*(1.0 - e_ice[4])*(TBD[4]+tau[4]*T_C))
        
        T31=Tv31*(math.sin(math.radians(0)))**2 + Th31*(math.cos(math.radians(0)))**2
        
        Th52=TBU[6]+tau[6]*((1.0 - c_ice)*emissivityh[6]*Ts+c_ice*e_ice[6]*Ti_amsr[6]+(1.0 - c_ice)*(1.0 - emissivityh[6])*\
             (T_BOmegaH[6]+tau[6]*T_C)+c_ice*(1.0 - e_ice[6])*(TBD[6]+tau[6]*T_C))

        Tv52=TBU[6]+tau[6]*((1.0 - c_ice)*emissivityv[6]*Ts+c_ice*e_ice[6]*Ti_amsr[6]+(1.0 - c_ice)*(1.0 - emissivityv[6])*\
             (T_BOmegaV[6]+tau[6]*T_C)+c_ice*(1.0 - e_ice[6])*(TBD[6]+tau[6]*T_C))
        
        T52=Tv52*(math.sin(math.radians(0)))**2 + Th52*(math.cos(math.radians(0)))**2
        
        Th6=TBU[0]+tau[0]*((1.0 - c_ice)*emissivityh[0]*Ts+c_ice*e_ice[0]*Ti_amsr[0]+(1.0 - c_ice)*(1.0 - emissivityh[0])*\
             (T_BOmegaH[0]+tau[0]*T_C)+c_ice*(1.0 - e_ice[0])*(TBD[0]+tau[0]*T_C))

        Tv6=TBU[0]+tau[0]*((1.0 - c_ice)*emissivityv[0]*Ts+c_ice*e_ice[0]*Ti_amsr[0]+(1.0 - c_ice)*(1.0 - emissivityv[0])*\
             (T_BOmegaV[0]+tau[0]*T_C)+c_ice*(1.0 - e_ice[0])*(TBD[0]+tau[0]*T_C))
        
        T6=Tv6*(math.sin(math.radians(0)))**2 + Th6*(math.cos(math.radians(0)))**2
        
        # Th89=TBU[7]+tau[7]*((1.0 - c_ice)*emissivityh[7]*Ts+c_ice*e_ice[7]*Ti_amsr[7]+(1.0 - c_ice)*(1.0 - emissivityh[7])*\
        #      (T_BOmegaH[7]+tau[7]*T_C)+c_ice*(1.0 - e_ice[7])*(TBD[7]+tau[7]*T_C))

        # Tv89=TBU[7]+tau[7]*((1.0 - c_ice)*emissivityv[7]*Ts+c_ice*e_ice[7]*Ti_amsr[7]+(1.0 - c_ice)*(1.0 - emissivityv[7])*\
        #      (T_BOmegaV[7]+tau[7]*T_C)+c_ice*(1.0 - e_ice[7])*(TBD[7]+tau[7]*T_C))
        
        # T89=Tv89*(math.sin(math.radians(0)))**2 + Th89*(math.cos(math.radians(0)))**2
        
        
        #channel 1 on scams 22.231H, channel 2 31.650H, channel 3 52.863H
        Tb=np.array([T6,T22,T31,T52])
    #return Tv, Th
    return Tb
def scams(V,W,L,Ta,Ts):
    """
    Calculate the brightness temperatures at different microwave frequencies,
    considering the impact of atmospheric and surface variables on satellite microwave measurements,
    made for NEMS 5 dataset
    
    Args:
    V (float): Columnar water vapor [mm].(tcwv)
    W (float): Wind speed at the surface (m/s).
    L (float): Columnar cloud liquid water [mm] (tcw)
    Ta (float): Ambient air temperature (degrees Celsius).
    Ts (float): Sea surface temperature (degrees Celsius).
    Ti_amsr (array8): Ice Temperature [K] 
    c_ice (float): Ice concentration, ranging from 0 (no ice) to 1 (full ice cover).
    e_ice (array): Ice emmisivity

    Returns:
    Tb (array): Brightness temperatures for channels 22.231V, 31.650V, and 52.863V.
    """
    Ti_amsrv = np.full(7, 260)
    Ti_amsrh = np.full(7, 260)
    theta = 0
    e_icev = np.full(7, 0.9)
    e_iceh = np.full(7, 0.9)
    # Frequency values for microwave measurements
    frequencies=np.array([6.93, 10.65, 18.70, 23.80, 36.50, 50.30, 52.80, 89.00])
    
    # Empirical coefficients for different calculations
    b0=np.array([239.50E+0,  239.51E+0,  240.24E+0,  241.69E+0,  239.45E+0,  242.10E+0,  245.87E+0,  242.58E+0])
    b1=np.array([213.92E-2,  225.19E-2,  298.88E-2,  310.32E-2,  254.41E-2,  229.17E-2,  250.61E-2,  302.33E-2])
    b2=np.array([-460.60E-4, -446.86E-4, -725.93E-4, -814.29E-4, -512.84E-4, -508.05E-4, -627.89E-4, -749.76E-4])
    b3=np.array([457.11E-6,  391.82E-6,  814.50E-6,  998.93E-6,  452.02E-6,  536.90E-6,  759.62E-6,  880.66E-6])
    b4=np.array([-16.84E-7,  -12.20E-7,  -36.07E-7,  -48.37E-7,  -14.36E-7,  -22.07E-7,  -36.06E-7,  -40.88E-7])
    b5=np.array([0.50E+0,     0.54E+0,    0.61E+0,    0.20E+0,    0.58E+0,    0.52E+0,    0.53E+0,    0.62E+0])
    b6=np.array([-0.11E+0,   -0.12E+0,   -0.16E+0,   -0.20E+0,   -0.57E+0,   -4.59E+0,  -12.52E+0,   -0.57E+0])
    b7=np.array([-0.21E-2,   -0.34E-2,   -1.69E-2,   -5.21E-2,   -2.38E-2,   -8.78E-2,  -23.26E-2,   -8.07E-2])
    ao1=np.array([8.34E-3,    9.08E-3,   12.15E-3,   15.75E-3,   40.06E-3,  353.72E-3, 1131.76E-3,   53.35E-3])
    ao2=np.array([-0.48E-4,  -0.47E-4,   -0.61E-4,   -0.87E-4,   -2.00E-4,  -13.79E-4,   -2.26E-4,   -1.18E-4])
    av1=np.array([0.07E-3,    0.18E-3,    1.73E-3,    5.14E-3,    1.88E-3,    2.91E-3,    3.17E-3,    8.78E-3])
    av2=np.array([0.00E-5,    0.00E-5,   -0.05E-5,    0.19E-5,    0.09E-5,    0.24E-5,    0.27E-5,    0.80E-5])

    aL1=np.array([0.0078, 0.0183, 0.0556, 0.0891,  0.2027,  0.3682,  0.4021,  0.9693])
    aL2=np.array([0.0303, 0.0298, 0.0288, 0.0281,  0.0261,  0.0236,  0.0231,  0.0146])
    aL3=np.array([0.0007, 0.0027, 0.0113, 0.0188,  0.0425,  0.0731,  0.0786,  0.1506])
    aL4=np.array([0.0000, 0.0060, 0.0040, 0.0020, -0.0020, -0.0020, -0.0020, -0.0020])
    aL5=np.array([1.2216, 1.1795, 1.0636, 1.0220,  0.9546,  0.8983,  0.8943,  0.7961])

    r0v=np.array([-0.27E-3,  -0.32E-3,  -0.49E-3,  -0.63E-3,  -1.01E-3, -1.20E-3, -1.23E-03, -1.53E-3])
    r0h=np.array([0.54E-3,   0.72E-3,   1.13E-3,   1.39E-3,   1.91E-3,  1.97E-3,  1.97E-03,  2.02E-3])
    r1v=np.array([-0.21E-4,  -0.29E-4,  -0.53E-4,  -0.70E-4,  -1.05E-4, -1.12E-4, -1.13E-04, -1.16E-4])
    r1h=np.array([0.32E-4,   0.44E-4,   0.70E-4,   0.85E-4,   1.12E-4,  1.18E-4,  1.19E-04,  1.30E-4])
    r2v=np.array([-2.10E-5,  -2.10E-5,  -2.10E-5,  -2.10E-5,  -2.10E-5, -2.10E-5, -2.10E-05, -2.10E-5])
    r2h=np.array([-25.26E-6, -28.94E-6, -36.90E-6, -41.95E-6, -54.51E-6, -5.50E-5, -5.50E-5,  -5.50E-5])
    r3v=np.array([0.00E-6,   0.08E-6,   0.31E-6,   0.41E-6,   0.45E-6,  0.35E-6,  0.32E-06, -0.09E-6])
    r3h=np.array([0.00E-6,  -0.02E-6,  -0.12E-6,  -0.20E-6,  -0.36E-6, -0.43E-6, -0.44E-06, -0.46E-6])

    m1v=np.array([0.00020, 0.00020, 0.00140, 0.00178, 0.00257, 0.00260, 0.00260, 0.00260])
    m1h=np.array([0.00200, 0.00200, 0.00293, 0.00308, 0.00329, 0.00330, 0.00330, 0.00330])
    m2v=np.array([0.00690, 0.00690, 0.00736, 0.00730, 0.00701, 0.00700, 0.00700, 0.00700])
    m2h=np.array([0.00600, 0.00600, 0.00656, 0.00660, 0.00660, 0.00660, 0.00660, 0.00660])
    
    # Arrays for storing calculated values
    TD=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    TU=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    AO=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    AV=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    AL=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    tau=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    TBU=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    TBD=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    llambda=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    epsilon=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], dtype=np.complex128)
    rho_H=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], dtype=np.complex128)
    rho_V=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0], dtype=np.complex128)
    R_0H=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    R_0V=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    R_geoH=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    R_geoV=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    F_H=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    F_V=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    R_H=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    R_V=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    OmegaH=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    OmegaV=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    T_BOmegaH=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    T_BOmegaV=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    emissivityh=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    emissivityv=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    term=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    Delta_S2=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    Tv=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    Th=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

    T_C=2.7

    if c_ice < 0.0: c_ice=0.0
    if c_ice > 1.0: c_ice=1.0

    Ts_mix=c_ice*Ta+(1.0-c_ice)*Ts

    Tl=(Ts_mix+273.0)/2.0

    #eq 27
    Tv=273.16+0.8337*V-3.029E-5*(V**3.33)
    if V > 48:  Tv=301.16
    G = 1.05*(Ts_mix-Tv)*(1-((Ts_mix-Tv)**2)/1200.0)
    if math.fabs(Ts_mix-Tv) > 20: G = (Ts_mix-Tv)*14/math.fabs(Ts_mix-Tv)

    epsilon_R=4.44 # this value is from wentz and meisner, 2000, p. 28
    s=35.0
    ny=0.012 # Klein and Swift is using 0.02 which is giving a higher epsilon_R (4.9)
    light_speed=3.00E10
    free_space_permittivity=8.854E-12
    #eq 43
    epsilon_S = (87.90*math.exp(-0.004585*(Ts-273.15)))*(math.exp(-3.45E-3*s+4.69E-6*s**2+1.36E-5*s*(Ts-273.15)))
    #eq 44
    lambda_R = (3.30*math.exp(-0.0346*(Ts-273.15)+0.00017*(Ts-273.15)**2))-(6.54E-3*(1-3.06E-2*(Ts-273.15)+2.0E-4*(Ts-273.15)**2)*s)
    #eq 41
    C=0.5536*s
    #eq 42
    delta_t=25.0-(Ts-273.15)
    #eq 40
    qsi=2.03E-2+1.27E-4*delta_t+2.46E-6*delta_t**2-C*(3.34E-5-4.60E-7*delta_t+4.60E-8*delta_t**2)
    #eq 39
    sigma=3.39E9*(C**0.892)*math.exp(-delta_t*qsi)

    for i in range(0,8):
        #eq26
        TD[i]=b0[i]+b1[i]*V+b2[i]*V**2+b3[i]*V**3+b4[i]*V**4+b5[i]*G
        TU[i]=TD[i]+b6[i]+b7[i]*V
        #eq 28
        AO[i]=ao1[i]+ao2[i]*(TD[i]-270.0)
        #eq 29
        AV[i]=av1[i]*V+av2[i]*V**2
        #eq 33
        AL[i]=aL1[i]*(1.0-aL2[i]*(Tl-283.0))*L
        #eq 22
        tau[i]=np.exp((-1.0/math.cos(math.radians(0)))*(AO[i]+AV[i]+AL[i])) 
        #eq 24
        TBU[i]=TU[i]*(1.0-tau[i])
        TBD[i]=TD[i]*(1.0-tau[i])

        llambda[i]=(light_speed/(frequencies[i]*1E9))

        #eq 35
        epsilon[i]=epsilon_R+((epsilon_S-epsilon_R)/(1.0+((cmath.sqrt(-1)*lambda_R)/llambda[i])**(1.0-ny)))-((2.0*cmath.sqrt(-1)*sigma*llambda[i])/light_speed)
        #eq.45
        rho_H[i]=(math.cos(math.radians(0))-cmath.sqrt(epsilon[i]-math.sin(math.radians(0))**2))/\
                 (math.cos(math.radians(0))+cmath.sqrt(epsilon[i]-math.sin(math.radians(0))**2))
        rho_V[i]=(epsilon[i]*math.cos(math.radians(0))-cmath.sqrt(epsilon[i]-math.sin(math.radians(0))**2))/\
                 (epsilon[i]*math.cos(math.radians(0))+cmath.sqrt(epsilon[i]-math.sin(math.radians(0))**2))
        #eq46
        R_0H[i]=np.absolute(rho_H[i])**2
        R_0V[i]=np.absolute(rho_V[i])**2+(4.887E-8-6.108E-8*(Ts-273.0)**3)

        #eq 57
        R_geoH[i]=R_0H[i]-(r0h[i]+r1h[i]*(0-53.0)+r2h[i]*(Ts-288.0)+r3h[i]*(0-53.0)*(Ts-288.0))*W
        R_geoV[i]=R_0V[i]-(r0v[i]+r1v[i]*(0-53.0)+r2v[i]*(Ts-288.0)+r3v[i]*(0-53.0)*(Ts-288.0))*W
        #eq.60
        W_1=7.0
        W_2=12.0
        if W<W_1: F_H[i]=m1h[i]*W
        elif W_1<W<W_2: F_H[i]=m1h[i]*W+0.5*(m2h[i]-m1h[i])*((W-W_1)**2)/(W_2-W_1)
        else: F_H[i]=m2h[i]*W-0.5*(m2h[i]-m1h[i])*(W_2+W_1)
        W_1=3.0
        W_2=12.0
        if W<W_1: F_V[i]=m1v[i]*W
        elif W_1<W<W_2: F_V[i]=m1v[i]*W+0.5*(m2v[i]-m1v[i])*((W-W_1)**2)/(W_2-W_1)
        else: F_V[i]=m2v[i]*W-0.5*(m2v[i]-m1v[i])*(W_2+W_1)

        R_H[i]=(1-F_H[i])*R_geoH[i]
        R_V[i]=(1-F_V[i])*R_geoV[i]

        emissivityh[i]=1-R_H[i]
        emissivityv[i]=1-R_V[i]

        if i >= 4: Delta_S2[i]=5.22E-3*W
        else: Delta_S2[i]=5.22E-3*(1-0.00748*(37.0-frequencies[i])**1.3)*W
        if Delta_S2[i]>0.069: Delta_S2[i]=0.069
        #eq.62
        term[i]=Delta_S2[i]-70.0*Delta_S2[i]**3
        OmegaH[i]=(6.2-0.001*(37.0-frequencies[i])**2)*term[i]*tau[i]**2.0
        OmegaV[i]=(2.5+0.018*(37.0-frequencies[i]))*term[i]*tau[i]**3.4
        #eq.61
        T_BOmegaH[i]=((1+OmegaH[i])*(1-tau[i])*(TD[i]-T_C)+T_C)*R_H[i] 
        T_BOmegaV[i]=((1+OmegaV[i])*(1-tau[i])*(TD[i]-T_C)+T_C)*R_V[i]
        
        
        Th22=TBU[3]+tau[3]*((1.0 - c_ice)*emissivityh[3]*Ts+c_ice*e_ice[3]*Ti_amsr[3]+(1.0 - c_ice)*(1.0 - emissivityh[3])*\
             (T_BOmegaH[3]+tau[3]*T_C)+c_ice*(1.0 - e_ice[3])*(TBD[3]+tau[3]*T_C))

        Tv22=TBU[3]+tau[3]*((1.0 - c_ice)*emissivityv[3]*Ts+c_ice*e_ice[3]*Ti_amsr[3]+(1.0 - c_ice)*(1.0 - emissivityv[3])*\
             (T_BOmegaV[3]+tau[3]*T_C)+c_ice*(1.0 - e_ice[3])*(TBD[3]+tau[3]*T_C))
        
        T22=Tv22*(math.sin(math.radians(0)))**2 + Th22*(math.cos(math.radians(0)))**2
        
        Th31=TBU[4]+tau[4]*((1.0 - c_ice)*emissivityh[4]*Ts+c_ice*e_ice[4]*Ti_amsr[4]+(1.0 - c_ice)*(1.0 - emissivityh[4])*\
             (T_BOmegaH[4]+tau[4]*T_C)+c_ice*(1.0 - e_ice[4])*(TBD[4]+tau[4]*T_C))

        Tv31=TBU[4]+tau[4]*((1.0 - c_ice)*emissivityv[4]*Ts+c_ice*e_ice[4]*Ti_amsr[4]+(1.0 - c_ice)*(1.0 - emissivityv[4])*\
             (T_BOmegaV[4]+tau[4]*T_C)+c_ice*(1.0 - e_ice[4])*(TBD[4]+tau[4]*T_C))
        
        T31=Tv31*(math.sin(math.radians(0)))**2 + Th31*(math.cos(math.radians(0)))**2
        
        Th52=TBU[6]+tau[6]*((1.0 - c_ice)*emissivityh[6]*Ts+c_ice*e_ice[6]*Ti_amsr[6]+(1.0 - c_ice)*(1.0 - emissivityh[6])*\
             (T_BOmegaH[6]+tau[6]*T_C)+c_ice*(1.0 - e_ice[6])*(TBD[6]+tau[6]*T_C))

        Tv52=TBU[6]+tau[6]*((1.0 - c_ice)*emissivityv[6]*Ts+c_ice*e_ice[6]*Ti_amsr[6]+(1.0 - c_ice)*(1.0 - emissivityv[6])*\
             (T_BOmegaV[6]+tau[6]*T_C)+c_ice*(1.0 - e_ice[6])*(TBD[6]+tau[6]*T_C))
        
        T52=Tv52*(math.sin(math.radians(0)))**2 + Th52*(math.cos(math.radians(0)))**2
        
        #channel 1 on scams 22.231H, channel 2 31.650H, channel 3 52.863H
        Tb=np.array([T22,T31])
    #return Tv, Th
    return Tb
def NEMSv2(V, W, L, Ta, Ts,c_ice):
    
    # NEMS freqensies
    frequencies = np.array([22.235, 31.400, 53.650, 54.900, 58.800])

    # Model Coefficients for the Atmosphere pr frequencies FOR AMSR!!!
    b0 = [241.69, 239.45, 245.87, 245.87, 245.87]
    b1 = [3.1032, 2.5441, 2.5061, 2.5061, 2.5061]
    b2 = [-0.081429, -0.051284, -0.062789, -0.062789, -0.062789]
    b3 = [0.00099893, 0.00045202, 0.00075962, 0.00075962, 0.00075962]
    b4 = [-4.837e-06, -1.436e-06, -3.606e-06, -3.606e-06, -3.606e-06]
    b5 = [0.20, 0.58, 0.53, 0.53, 0.53]
    b6 = [-0.20, -0.57, -12.52, -12.52, -12.52]
    b7 = [-0.0521, -0.0238, -0.2326, -0.2326, -0.2326]
    
    # RMS Error in Oxygen and Water Vapor Absorption Approximation FOR AMSR!!!
    ao1 = [0.01575, 0.04006, 1.13176, 1.13176, 1.13176]
    ao2 = [-0.000087, -0.000200, -0.000226, -0.000226, -0.000226]
    av1 = [0.00514, 0.00188, 0.00317, 0.00317, 0.00317]
    av2 = [0.19e-5, 0.09e-5, 0.27e-5, 0.27e-5, 0.27e-5]

    # Coefficients for Rayleigh Absorption and Mie ScatterinG FOR AMSR
    aL1 = [0.0891, 0.2027, 0.4021, 0.4021, 0.4021]
    aL2 = [0.0281, 0.0261, 0.0231, 0.0231, 0.0231]

    # Model Coefficients for Geometric Optics for AMSR
    r0v = [-0.00063, -0.00101, -0.00123, -0.00123, -0.00123]
    r0h = [0.00139, 0.00191, 0.00197, 0.00197, 0.00197]
    r1v = [-0.000070, -0.000105, -0.000113, -0.000113, -0.000113]
    r1h = [0.000085, 0.000112, 0.000119, 0.000119, 0.000119]
    r2v = [-2.1e-05, -2.1e-05, -2.1e-05, -2.1e-05, -2.1e-05]
    r2h = [-4.195e-05, -5.451e-05, -5.5e-05, -5.5e-05, -5.5e-05]
    r3v = [0.41e-06, 0.45e-06, 0.32e-06, 0.32e-06, 0.32e-06]
    r3h = [-0.20e-06, -0.36e-06, -0.44e-06, -0.44e-06, -0.44e-06]
    
        
    # The coefficients m1 and m2. Units are s/m. 
    m1v = [0.00178, 0.00257, 0.00260, 0.00260, 0.00260]
    m1h = [0.00308, 0.00329, 0.00330, 0.00330, 0.00330]
    m2v = [0.00730, 0.00701, 0.00700, 0.00700, 0.00700]
    m2h = [0.00660, 0.00660, 0.00660, 0.00660, 0.00660]
    
    # Ice temperatures for freqs
    Ti_amsrv = np.full(7, 260)
    Ti_amsrh = np.full(7, 260)
    theta = 0
    e_icev = np.full(7, 0.9)
    e_iceh = np.full(7, 0.9)
    
    # --------------------------------------
    # Here the real simulation begins
    T_C = 2.7
    c_ice = max(0.0, min(c_ice, 1.0))
    Ts_mix = c_ice*Ta + (1-c_ice)*Ts
    Tl = (Ts_mix + 273.0)/2.0
    
    #eq 27 !!
    Tv_air = 273.16 + 0.8337*V - 3.029E-5*V**3.33
    if V > 48: Tv_air = 301.16
    G = 1.05*(Ts_mix-Tv_air)*(1-((Ts_mix-Tv_air)**2)/1200.0)
    if abs(Ts_mix-Tv_air) > 20: G = (Ts_mix-Tv_air)*14/abs(Ts_mix-Tv_air)

    epsilon_R = 4.44 # this value is from wentz and meisner, 2000, p. 28
    s = 35.0
    ny = 0.012  # Klein and Swift is using 0.02 which is giving a higher epsilon_R (4.9)
    light_speed = 3.0E10

    #eq 43
    epsilon_S = (87.90*np.exp(-0.004585*(Ts-273.15)))*(np.exp(-3.45E-3*s+4.69E-6*s**2+1.36E-5*s*(Ts-273.15)))
    
    #eq 44
    lambda_R = (3.30*np.exp(-0.0346*(Ts-273.15)+0.00017*(Ts-273.15)**2))-(6.54E-3*(1-3.06E-2*(Ts-273.15)+2.0E-4*(Ts-273.15)**2)*s)
    #eq 41
    C = 0.5536*s
    #eq 42
    delta_t = 25.0-(Ts-273.15)
    #eq 40
    qsi = 2.03E-2+1.27E-4*delta_t+2.46E-6*delta_t**2-C*(3.34E-5-4.60E-7*delta_t+4.60E-8*delta_t**2)
    #eq 39
    sigma = 3.39E9*(C**0.892)*np.exp(-delta_t*qsi)

    Tb = np.zeros(5)

    for i in range(5):
        #eq26
        # Downwelling radiation
        TD = b0[i]+b1[i]*V+b2[i]*V**2+b3[i]*V**3+b4[i]*V**4+b5[i]*G
        # Upwelling radiation
        TU = TD + b6[i]+b7[i]*V
        #eq 28
        AO = ao1[i]+ao2[i]*(TD-270.0)
        #eq 29
        AV = av1[i]*V + av2[i]*V**2
        #eq 33
        AL = aL1[i]*(1.0 - aL2[i]*(Tl-283.0))*L
        #eq 22
        tau = np.exp((-1.0/np.cos(np.radians(theta)))*(AO+AV+AL))
        #eq 24
        TBU = TU*(1.0-tau)
        TBD = TD*(1.0-tau)

        llambda = light_speed/(frequencies[i]*1e9)
        
        #eq 35
        epsilon = epsilon_R + ((epsilon_S-epsilon_R)/(1 + (1j*lambda_R/llambda)**(1-ny))) - (2*1j*sigma*llambda/light_speed)
        
        #eq.45
        rho_H = (np.cos(np.radians(theta)) - np.sqrt(epsilon - np.sin(np.radians(theta))**2)) / \
                (np.cos(np.radians(theta)) + np.sqrt(epsilon - np.sin(np.radians(theta))**2))
        rho_V = (epsilon*np.cos(np.radians(theta)) - np.sqrt(epsilon - np.sin(np.radians(theta))**2)) / \
                (epsilon*np.cos(np.radians(theta)) + np.sqrt(epsilon - np.sin(np.radians(theta))**2))
        
        #eq46
        R_0H = abs(rho_H)**2
        R_0V = abs(rho_V)**2 + (4.887E-8 - 6.108E-8*(Ts-273.0)**3)
        
        #eq 57
        R_geoH = R_0H - (r0h[i]+r1h[i]*(theta-53.0)+r2h[i]*(Ts-288.0)+r3h[i]*(theta-53.0)*(Ts-288.0))*W
        R_geoV = R_0V - (r0v[i]+r1v[i]*(theta-53.0)+r2v[i]*(Ts-288.0)+r3v[i]*(theta-53.0)*(Ts-288.0))*W
        
        # Check article!!
        #eq.60
        W1, W2 = 7.0, 12.0
        
        if W<W1: 
            F_H = m1h[i]*W
        elif W<W2:
            F_H = m1h[i]*W + 0.5*(m2h[i]-m1h[i])*((W-W1)**2)/(W2-W1)
        else: 
            F_H = m2h[i]*W - 0.5*(m2h[i]-m1h[i])*(W2+W1)

        W1, W2 = 3.0, 12.0
        
        if W<W1:
            F_V = m1v[i]*W
        elif W<W2:
            F_V = m1v[i]*W + 0.5*(m2v[i]-m1v[i])*((W-W1)**2)/(W2-W1)
        else:
            F_V = m2v[i]*W - 0.5*(m2v[i]-m1v[i])*(W2+W1)

        R_H = (1-F_H)*R_geoH
        R_V = (1-F_V)*R_geoV
        emissivityh = 1-R_H
        emissivityv = 1-R_V

        Delta_S2 = 5.22E-3*W if i>=4 else 5.22E-3*(1-0.00748*(37-frequencies[i])**1.3)*W
        Delta_S2 = min(Delta_S2,0.069)
        
        #eq.62
        term = Delta_S2 - 70.0*Delta_S2**3
        OmegaH = (6.2-0.001*(37-frequencies[i])**2)*term*tau**2
        OmegaV = (2.5+0.018*(37-frequencies[i]))*term*tau**3.4
        #eq.61
        T_BOmegaH = ((1+OmegaH)*(1-tau)*(TD-T_C)+T_C)*R_H
        T_BOmegaV = ((1+OmegaV)*(1-tau)*(TD-T_C)+T_C)*R_V

        Th = TBU + tau*((1-c_ice)*emissivityh*Ts + c_ice*e_iceh[i]*Ti_amsrh[i] + (1-c_ice)*(1-emissivityh)*(T_BOmegaH+tau*T_C) + c_ice*(1-e_iceh[i])*(TBD+tau*T_C))
        Tv = TBU + tau*((1-c_ice)*emissivityv*Ts + c_ice*e_icev[i]*Ti_amsrv[i] + (1-c_ice)*(1-emissivityv)*(T_BOmegaV+tau*T_C) + c_ice*(1-e_icev[i])*(TBD+tau*T_C))

        Tb[i] = Tv*(np.sin(np.radians(theta)))**2 + Th*(np.cos(np.radians(theta)))**2

    return Tb