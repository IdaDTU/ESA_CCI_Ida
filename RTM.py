#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""---------------------------------------------------------------------------
AMSR Ocean Algorithm; Frank J. Wentz, Thomas Meissner; Remote
Sensing Systems; Version 2; November 2; 2000. Modified to NIMBUS 6 SCAMS geometry.
NIMBUS 6 SCAMS Scanning Microwave Spectrometer 145 km spatial resolution 2400 km swath
Tb=scams(V,W,L,Ta,Ts,theta,Ti_amsrv,Ti_amsrh,c_ice,e_icev,e_iceh)
V: columnar water vapor [mm]
W: windspeed over water [m/s]
L: columnar cloud liquid water [mm]
Ts: sea surface temperature [K]
theta: incidence angle [deg]
Ti: ice surface temperature [K]
c_ice: ice concentration [0-1]
e_ice: ice emissivity [0-1]
-------------------------------------------------------------------------------"""

import numpy as np
from numpy import lib
import sys, os, string, math, cmath


def NEMSv3(V,W,L,Ta,Ts,c_ice):
    
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
    Ti_amsrv = np.full(9, 260)
    Ti_amsrh = np.full(9, 260)
    theta = 0
    e_icev = np.full(9, 0.9)
    e_iceh = np.full(9, 0.9)
    
    # --------------------------------------
    # Here the real simulation begins
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


    for i in range(0,5):
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
        tau[i]=math.exp((-1.0/math.cos(math.radians(theta)))*(AO[i]+AV[i]+AL[i])) 
        #eq 24
        TBU[i]=TU[i]*(1.0-tau[i])
        TBD[i]=TD[i]*(1.0-tau[i])

        llambda[i]=(light_speed/(frequencies[i]*1E9))

        #eq 35
        epsilon[i]=epsilon_R+((epsilon_S-epsilon_R)/(1.0+((cmath.sqrt(-1)*lambda_R)/llambda[i])**(1.0-ny)))-((2.0*cmath.sqrt(-1)*sigma*llambda[i])/light_speed)
        #eq.45
        rho_H[i]=(math.cos(math.radians(theta))-cmath.sqrt(epsilon[i]-math.sin(math.radians(theta))**2))/\
                 (math.cos(math.radians(theta))+cmath.sqrt(epsilon[i]-math.sin(math.radians(theta))**2))
        rho_V[i]=(epsilon[i]*math.cos(math.radians(theta))-cmath.sqrt(epsilon[i]-math.sin(math.radians(theta))**2))/\
                 (epsilon[i]*math.cos(math.radians(theta))+cmath.sqrt(epsilon[i]-math.sin(math.radians(theta))**2))
        #eq46
        R_0H[i]=np.absolute(rho_H[i])**2
        R_0V[i]=np.absolute(rho_V[i])**2+(4.887E-8-6.108E-8*(Ts-273.0)**3)

        #eq 57
        R_geoH[i]=R_0H[i]-(r0h[i]+r1h[i]*(theta-53.0)+r2h[i]*(Ts-288.0)+r3h[i]*(theta-53.0)*(Ts-288.0))*W
        R_geoV[i]=R_0V[i]-(r0v[i]+r1v[i]*(theta-53.0)+r2v[i]*(Ts-288.0)+r3v[i]*(theta-53.0)*(Ts-288.0))*W
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
        
        # Something is going on here!! 
        if i >= 2:  # Adapted to NEMS now, was 4 before
            Delta_S2[i]=5.22E-3*W
        else:
            Delta_S2[i]=5.22E-3*(1-0.00748*(37.0-frequencies[i])**1.3)*W
        if Delta_S2[i]>0.069: 
            Delta_S2[i]=0.069
        
        #eq.62
        term[i]=Delta_S2[i]-70.0*Delta_S2[i]**3
        OmegaH[i]=(6.2-0.001*(37.0-frequencies[i])**2)*term[i]*tau[i]**2.0
        OmegaV[i]=(2.5+0.018*(37.0-frequencies[i]))*term[i]*tau[i]**3.4
        #eq.61
        T_BOmegaH[i]=((1+OmegaH[i])*(1-tau[i])*(TD[i]-T_C)+T_C)*R_H[i] 
        T_BOmegaV[i]=((1+OmegaV[i])*(1-tau[i])*(TD[i]-T_C)+T_C)*R_V[i]
        
    #ch1
    Th22=TBU[0]+tau[0]*((1.0 - c_ice)*emissivityh[0]*Ts+c_ice*e_iceh[0]*Ti_amsrh[0]+(1.0 - c_ice)*(1.0 - emissivityh[0])*\
             (T_BOmegaH[0]+tau[0]*T_C)+c_ice*(1.0 - e_iceh[0])*(TBD[0]+tau[0]*T_C))

    Tv22=TBU[0]+tau[0]*((1.0 - c_ice)*emissivityv[0]*Ts+c_ice*e_icev[0]*Ti_amsrv[0]+(1.0 - c_ice)*(1.0 - emissivityv[0])*\
         (T_BOmegaV[0]+tau[0]*T_C)+c_ice*(1.0 - e_icev[0])*(TBD[0]+tau[0]*T_C))
    
    T22=Tv22*(math.sin(math.radians(theta)))**2 + Th22*(math.cos(math.radians(theta)))**2
        
        
    # ch2
    Th31=TBU[1]+tau[1]*((1.0 - c_ice)*emissivityh[1]*Ts+c_ice*e_iceh[1]*Ti_amsrh[1]+(1.0 - c_ice)*(1.0 - emissivityh[1])*\
         (T_BOmegaH[1]+tau[1]*T_C)+c_ice*(1.0 - e_iceh[1])*(TBD[1]+tau[1]*T_C))

    Tv31=TBU[1]+tau[1]*((1.0 - c_ice)*emissivityv[1]*Ts+c_ice*e_icev[1]*Ti_amsrv[1]+(1.0 - c_ice)*(1.0 - emissivityv[1])*\
         (T_BOmegaV[1]+tau[1]*T_C)+c_ice*(1.0 - e_icev[1])*(TBD[1]+tau[1]*T_C))
        
    T31=Tv31*(math.sin(math.radians(theta)))**2 + Th31*(math.cos(math.radians(theta)))**2
        
    # ch3
    Th53=TBU[2]+tau[2]*((1.0 - c_ice)*emissivityh[2]*Ts+c_ice*e_iceh[2]*Ti_amsrh[2]+(1.0 - c_ice)*(1.0 - emissivityh[2])*\
         (T_BOmegaH[2]+tau[2]*T_C)+c_ice*(1.0 - e_iceh[2])*(TBD[2]+tau[2]*T_C))

    Tv53=TBU[2]+tau[2]*((1.0 - c_ice)*emissivityv[2]*Ts+c_ice*e_icev[2]*Ti_amsrv[2]+(1.0 - c_ice)*(1.0 - emissivityv[2])*\
         (T_BOmegaV[2]+tau[2]*T_C)+c_ice*(1.0 - e_icev[2])*(TBD[2]+tau[2]*T_C))
        
    T53=Tv53*(math.sin(math.radians(theta)))**2 + Th53*(math.cos(math.radians(theta)))**2
        
    # ch4
    Th54=TBU[3]+tau[3]*((1.0 - c_ice)*emissivityh[3]*Ts+c_ice*e_iceh[3]*Ti_amsrh[3]+(1.0 - c_ice)*(1.0 - emissivityh[3])*\
         (T_BOmegaH[3]+tau[3]*T_C)+c_ice*(1.0 - e_iceh[3])*(TBD[3]+tau[3]*T_C))

    Tv54=TBU[3]+tau[3]*((1.0 - c_ice)*emissivityv[3]*Ts+c_ice*e_icev[3]*Ti_amsrv[3]+(1.0 - c_ice)*(1.0 - emissivityv[3])*\
        (T_BOmegaV[3]+tau[3]*T_C)+c_ice*(1.0 - e_icev[3])*(TBD[3]+tau[3]*T_C))
        
    T54=Tv54*(math.sin(math.radians(theta)))**2 + Th54*(math.cos(math.radians(theta)))**2
        
    # ch5
    Th58=TBU[4]+tau[4]*((1.0 - c_ice)*emissivityh[4]*Ts+c_ice*e_iceh[4]*Ti_amsrh[4]+(1.0 - c_ice)*(1.0 - emissivityh[4])*\
         (T_BOmegaH[4]+tau[4]*T_C)+c_ice*(1.0 - e_iceh[4])*(TBD[4]+tau[4]*T_C))

    Tv58=TBU[4]+tau[4]*((1.0 - c_ice)*emissivityv[4]*Ts+c_ice*e_icev[4]*Ti_amsrv[4]+(1.0 - c_ice)*(1.0 - emissivityv[4])*\
         (T_BOmegaV[4]+tau[4]*T_C)+c_ice*(1.0 - e_icev[4])*(TBD[4]+tau[4]*T_C))
        
    T58=Tv58*(math.sin(math.radians(theta)))**2 + Th58*(math.cos(math.radians(theta)))**2

    # frequencies = np.array([22.235, 31.400, 53.650, 54.900, 58.800])
    Tb=np.array([T22, T31,T53,T54,T58])
    #return Tv, Th
    return Tb