import numpy as np
import pandas as pd
import os
from math import pi, exp, sqrt
from scipy.interpolate import interp1d
from copy import deepcopy
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from numba import jit
from scipy.linalg import solve_banded
from scipy.stats.stats import pearsonr
from scipy import integrate
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from math import pi, exp, sqrt, log, atan, sin, radians, nan, isinf, ceil
from scipy.interpolate import interp1d

def eddy_diffusivity(rho, depth, g, rho_0, ice, area, T, diff):
    km = 1.4 * 10**(-7)
    
    rho = np.array(rho)
    
    buoy = np.ones(len(depth)) * 7e-5
    buoy[:-1] = np.abs(rho[1:] - rho[:-1]) / (depth[1:] - depth[:-1]) * g / rho_0
    buoy[-1] = buoy[-2]
        
    low_values_flags = buoy < 7e-5  # Where values are low
    buoy[low_values_flags] = 7e-5
    
    if ice == True:
      ak = 0.000898
    else:
      ak = 0.00706 *( max(area)/1E6)**(0.56)
    
    kz = ak * (buoy)**(-0.43)
    
        
    if (np.mean(diff) == 0.0):
        weight = 1
    else:
        weight = 0.5
        
    kz = weight * kz + (1 - weight) * diff

    
    return(kz + km)

def eddy_diffusivity_hendersonSellers(rho, depth, g, rho_0, ice, area, U10, latitude, T, diff, Cd, km, weight_kz, k0):
    k = 0.4
    Pr = 1.0
    z0 = 0.0002
    # 1.4 * 10**(-7)
    f = 1 * 10 **(-4)
    xi = 1/3
    kullenberg = 2 * 10**(-2)
    rho_a = 1.2


    
    U2 = U10 * 10
    U2 = U10 * (log((2 - 1e-5)/z0)) / (log((10 - 1e-5)/z0))
    
    if U2 < 2.2:
        Cd = 1.08 * U2**(-0.15)* 10**(-3)
    elif 2.2 <= U2 < 5.0:
        Cd = (0.771 + 0.858 * U2**(-0.15)) *10**(-3)
    elif 5.0 <= U2 < 8.0:
        Cd = (0.867 + 0.0667 * U2**(-0.15)) * 10**(-3)
    elif 8.0 <= U2 < 25:
        Cd = (1.2 + 0.025 * U2**(-0.15)) * 10**(-3)
    elif 25 <= U2 < 50:
        Cd = 0.073 * U2**(-0.15) * 10**(-3)
        
    w_star = Cd * U2
    k_star = 6.6 * (sin(radians(latitude)))**(1/2) * U2**(-1.84)
    

    buoy = np.ones(len(depth)) * 7e-5
    buoy[:-1] = np.abs(rho[1:] - rho[:-1]) / (depth[1:] - depth[:-1]) * g / rho[:-1] #rho_0
    buoy[-1] = buoy[-2]
        
    low_values_flags = buoy < 7e-5  # Where values are low
    buoy[low_values_flags] = 7e-5
    
    s_bg = 2 * 10**(-7)
    s_seiche = 0.7 * buoy
    

    Ri = (-1 + (1 + 40 * (np.array(buoy) * k**2 * np.array(depth)**2) / 
               (w_star**2 * np.exp(-2 * k_star * np.array(depth))))**(1/2)) / 20
    
    kz = (k * w_star * np.array(depth)) / (Pr * (1 + 37 * np.array(Ri)**2)) * np.exp(-k_star * np.array(depth))
    
    tau_w = rho_a * Cd * U2**2
    u_star = sqrt(tau_w / rho_0)
    H_ekman = 0.4 * u_star / f
    
    e_w = xi * sqrt(Cd) * U2
    W_eff = e_w / (xi * sqrt(Cd))
    kz_ekman = 1/f * (rho_a / rho_0 * Cd / kullenberg)**2 * W_eff**2
    
    kz_old = kz
    
    # if (np.mean(T) <= 5):
        # kz = kz * 1000
    # Hongping Gu et al. (2015). Climate Change
    # LST = T[0]
    # if (LST > 4):
    #     kz = kz * 10**2
    # elif (LST > 0) & (LST <= 4):
    #     kz = kz * 10**4
    # elif LST <= 0:
    #     kz = kz * 0
    

    if (np.mean(diff) == 0.0):
        weight = 1
    else:
        weight = weight_kz

    kz = weight * kz + (1 - weight) * diff

    return(kz +  km)

## this is our attempt for turbulence closure, estimating eddy diffusivity
def eddy_diffusivity_munkAnderson(rho, depth, g, rho_0, ice, area, U10, latitude, Cd, T, diff, k0, km, weight_kz):
    k = 0.4
    Pr = 1.0
    z0 = 0.0002
    km = 1.4 * 10**(-7)
    rho_a = 1.2
    alpha = 10/3
    beta = 3/2
    f = 1 * 10 **(-4)
    xi = 1/3
    kullenberg = 2 * 10**(-2)
    
    U2 = U10 * (log((2 - 1e-5)/z0)) / (log((10 - 1e-5)/z0))
    U2 = U10
    
    if U2 < 2.2:
        Cd = 1.08 * U2**(-0.15)* 10**(-3)
    elif 2.2 <= U2 < 5.0:
        Cd = (0.771 + 0.858 * U2**(-0.15)) *10**(-3)
    elif 5.0 <= U2 < 8.0:
        Cd = (0.867 + 0.0667 * U2**(-0.15)) * 10**(-3)
    elif 8.0 <= U2 < 25:
        Cd = (1.2 + 0.025 * U2**(-0.15)) * 10**(-3)
    elif 25 <= U2 < 50:
        Cd = 0.073 * U2**(-0.15) * 10**(-3)
    
    w_star = sqrt(rho_a / rho[0] * Cd * U2**2)
    k_star = 0.51 * (sin(radians(latitude))) / U2
    
    
    
    buoy = np.ones(len(depth)) * 7e-5
    buoy[:-1] = np.abs(rho[1:] - rho[:-1]) / (depth[1:] - depth[:-1]) * g / rho_0
    buoy[-1] = buoy[-2]
        
    low_values_flags = buoy < 7e-5  # Where values are low
    buoy[low_values_flags] = 7e-5
    
    s_bg = 2 * 10**(-7)
    s_seiche = 0.7 * buoy

    s_wall = (w_star / (k * np.array(depth)) * np.exp(k_star * np.array(depth)))**2
    s_wall = w_star/ (k * np.array(depth) *np.array(rho))
    
    
    X_HS = np.array(buoy)/(s_wall**2 + s_bg + s_seiche)
    Ri=(-1+(1+40*X_HS)**0.5)/20
    

    f_HS = (1.0 / (1 + alpha * Ri)**beta)
    f_HS[Ri == 0] = 1
    
    kz = (k * w_star * np.array(depth)) * np.exp(-k_star * np.array(depth)) * f_HS
    
    kz[0] = kz[1]
    # modify according to Ekman layer depth
    
    tau_w = rho_a * Cd * U2**2
    u_star = sqrt(tau_w / rho_0)
    H_ekman = 0.4 * u_star / f
    
    e_w = xi * sqrt(Cd) * U2
    W_eff = e_w / (xi * sqrt(Cd))
    kz_ekman = 1/f * (rho_a / rho_0 * Cd / kullenberg)**2 * W_eff**2
    

    if (np.mean(T) <= 5):
        kz = kz * 1000
    
    if (np.mean(diff) == 0.0):
        weight = 1
    else:
        weight = 0.5
        
    kz = weight * kz + (1 - weight) * diff

    return(kz +  km)

def eddy_diffusivity_pacanowskiPhilander(rho, depth, g, rho_0, ice, area, U10, latitude, T, diff, Cd, km, weight_kz, k0):
    k = 0.4
    Pr = 1.0
    z0 = 0.0002
    # 1.4 * 10**(-7)
    f = 1 * 10 **(-4)
    xi = 1/3
    kullenberg = 2 * 10**(-2)
    rho_a = 1.2
    K0 = k0 # 10**(-2)
    Kb = 10**(-7)


    
    U2 = U10 * 10
    U2 = U10 * (log((2 - 1e-5)/z0)) / (log((10 - 1e-5)/z0))
    
    w_star = Cd * U2
    k_star = 6.6 * (sin(radians(latitude)))**(1/2) * U2**(-1.84)
    

    buoy = np.ones(len(depth)) * 7e-5
    buoy[:-1] = np.abs(rho[1:] - rho[:-1]) / (depth[1:] - depth[:-1]) * g / rho_0
    buoy[-1] = buoy[-2]
        
    low_values_flags = buoy < 7e-5  # Wherevalues are low
    buoy[low_values_flags] = 7e-5
    
    s_bg = 2 * 10**(-7)
    s_seiche = 0.7 * buoy

    Ri = (-1 + (1 + 40 * (np.array(buoy) * k**2 * np.array(depth)**2) / 
               (w_star**2 * np.exp(-2 * k_star * np.array(depth))))**(1/2)) / 20
    
    kz = K0 / (1 + 5 * Ri)**2 + Kb
    
    kz[0] = kz[1] * 1.1
    

    
    if (np.mean(diff) == 0.0):
        weight = 1
    else:
        weight = weight_kz
        
    kz = weight * kz + (1 - weight) * diff

    return(kz)


## function to calculate density from temperature
def calc_dens(wtemp):
    dens = (999.842594 + (6.793952 * 1e-2 * wtemp) - (9.095290 * 1e-3 *wtemp**2) +
      (1.001685 * 1e-4 * wtemp**3) - (1.120083 * 1e-6* wtemp**4) + 
      (6.536336 * 1e-9 * wtemp**5))
    return dens


## lake configurations
zmax = 25 # maximum lake depth
nx = 25 * 2 # number of layers we will have
dt = 3600 # 24 hours times 60 min/hour times 60 seconds/min
dx = zmax/nx # spatial step

area = np.array([39850000., 37925000., 36000000., 35000000., 34000000., 33050000.,
       32100000., 31200000., 30300000., 29500000., 28700000., 28250000.,
       27800000., 27400000., 27000000., 26500000., 26000000., 25500000.,
       25000000., 24500000., 24000000., 23350000., 22700000., 21900000.,
       21100000., 20500000., 19900000., 19150000., 18400000., 18000000.,
       17600000., 17000000., 16400000., 15800000., 15200000., 14450000.,
       13700000., 12500000., 11300000.,  9950000.,  8600000.,  7250000.,
        5900000.,  4700000.,  3500000.,  2550000.,  1600000.,   850000.,
         100000.,    50000.])
depth = np.array([ 0.25,  0.75,  1.25,  1.75,  2.25,  2.75,  3.25,  3.75,  4.25,
        4.75,  5.25,  5.75,  6.25,  6.75,  7.25,  7.75,  8.25,  8.75,
        9.25,  9.75, 10.25, 10.75, 11.25, 11.75, 12.25, 12.75, 13.25,
       13.75, 14.25, 14.75, 15.25, 15.75, 16.25, 16.75, 17.25, 17.75,
       18.25, 18.75, 19.25, 19.75, 20.25, 20.75, 21.25, 21.75, 22.25,
       22.75, 23.25, 23.75, 24.25, 24.75])
g = 9.81
ice = False
Uw_n = 2
km = 1.4 * 10**(-7) 
k0 = 1 * 10**(-2) 
weight_kz = 1 
Cd = 0.0013



### START OF IMPLICIT SCHEME
u =  np.linspace(25, 12, nx)
u = np.concatenate([np.array([27,29.3]), np.linspace(25, 24, 20),np.linspace(24, 15, 5), np.linspace(15,13,23)])

un = u 
dens_u_n2 = calc_dens(un)
kz = u * 0.01+11111
diffusion_method = 'hendersonSellers'

if diffusion_method == 'hendersonSellers':
    kzn = eddy_diffusivity_hendersonSellers(dens_u_n2, depth, g, np.mean(dens_u_n2) , ice, area, Uw_n,  43.100948, u, kz, Cd, km, weight_kz, k0) 
elif diffusion_method == 'munkAnderson':
    kzn = eddy_diffusivity_munkAnderson(dens_u_n2, depth, g, np.mean(dens_u_n2) , ice, area, Uw_n,  43.100948, Cd, u, kz,  km, weight_kz, k0) 
elif diffusion_method == 'hondzoStefan':
    kzn = eddy_diffusivity(dens_u_n2, depth, g, np.mean(dens_u_n2) , ice, area, u, kz) / 86400
elif diffusion_method == 'pacanowskiPhilander':
    kzn = eddy_diffusivity_pacanowskiPhilander(dens_u_n2, depth, g, np.mean(dens_u_n2) , ice, area, Uw_n,  43.100948, u, kz, Cd, km, weight_kz, k0) 
        
kzn = kzn *100
print(max(kzn))
print(min(kzn))

# IMPLEMENTATION OF CRANK-NICHOLSON SCHEME
j = len(un)
y = np.zeros((len(un), len(un)))

area_diff = np.ones(len(depth)) * 0
kzn_diff = np.ones(len(depth)) * 0
for k in range(1, len(area_diff)-1):
    area_diff[k] = area[k + 1] - area[k - 1]  
    kzn_diff[k] = kzn[k + 1] - kzn[k - 1]  
        
kzn_diff[0] =  2 * (kzn[1] - kzn[0]) 
kzn_diff[len(depth)-1] =  2 * (kzn[len(depth)-1] - kzn[len(depth)-2]) 
        
area_diff[0] =  2 * (area[1] - area[0]) 
area_diff[len(depth)-1] =  2 * (area_diff[len(depth)-1] - area_diff[len(depth)-2]) 
        
        
        

alpha = ( dt) / (2 * area * dx**2)
print(max(alpha))

if max(alpha) > 1:
    print('Warning: alpha > 1')
    print("Warning: ",max(alpha)," > 1")
            
    if all(alpha > 1):
        alpha = alpha
    else:
        alpha[alpha > 1] = max(alpha[alpha < 1])
            

az = (alpha * kzn_diff * area) / 4 + (alpha * kzn * area_diff) / 4 - alpha * kzn * area   # subdiagonal
bz = (1 + 2 * alpha * area * kzn) # diagonal
cz = (- alpha * kzn_diff * area) / 4 - (alpha * kzn * area_diff) / 4 - alpha * kzn * area # superdiagonal
        
        
        
bz[0] = 1
bz[len(bz)-1] = 1
cz[0] = 0
        
#az =  np.delete(az,0)
#cz =  np.delete(cz,len(cz)-1)
        
# tridiagonal matrix
for k in range(j-1):
    y[k][k] = bz[k]
    y[k][k+1] = cz[k]
    y[k+1][k] = az[k]
        

y[j-1, j-2] = 0 
y[j-1, j-1] = 1
        
mn = un * 0.0    
mn[0] = un[0]
# mn[0] = - az[0] * un[0+1] + (1 - 2 * alpha[0] * area[0] * kzn[0]) * un[0] - cz[0] * un[0+1]
mn[-1] = un[-1]
# mn[-1] = - az[-1] * un[-1-1] + (1 - 2 * alpha[-1] * area[-1] * kzn[-1]) * un[-1] - cz[-1] * un[-1-1]
        
for k in range(1,j-1):
    mn[k] = - az[k] * un[k-1] + (1 - 2 * alpha[k] * area[k] * kzn[k]) * un[k] - cz[k] * un[k+1]


# DERIVED TEMPERATURE OUTPUT FOR NEXT MODULE
u = np.linalg.solve(y, mn)
        
plt.plot(un, depth)
plt.plot(u, depth, color ='red')
plt.gca().invert_yaxis()
plt.show()
