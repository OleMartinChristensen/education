#%%
import numpy as np
#%%

def calc_tau(T,theta=0):
    return -np.log(T)/np.cos(theta)

def calc_rho_n(T,sigma,l):
    tau = calc_tau(T)
    n = tau/(sigma*l*100)
    return n

def cal_rho(n,M):
    rho = n/Na*M 
    return rho

def A_cylinder(d):
    r = d/2
    return np.pi*r**2

def calc_flux(rho,A,v):
    A_cm = A*10000
    v_cm = v*100

    return rho*A_cm*v_cm #kg/cm**3 * m * m/s = kg/s

T = 0.001
sigma = 1e-21 #cm2/molec
Na = 6.0221408e+23 #mol-1
M = 12+4*1 #g/mol
theta = np.deg2rad(60)
d = 100
l = d/np.cos(theta)

n = calc_rho_n(T,sigma,l) #molec/cm3
rho = cal_rho(n,M) #g/cm^3

A = A_cylinder(d) #m2
v = 5 #m/s
F = calc_flux(rho*1e-3,A,v) #kg/s
print(F)
# %%
