import numpy as np
from matplotlib import pyplot as plt
## Task 1

def R_hw(f):
    R = 0.7363 - 0.001967*f - 1.4e-5*f**2 + 1.205e-7*f**3
    return R

def R_vw(f):
    R = 0.5419 - 0.002863*f - 8.664e-6*f**2 + 1.199e-7*f**3
    return R

def R_hi(f):
    R = 0.1555
    return R

def R_vi(f):
    R = 0.0242
    return R

def get_reflectivity(f,pol,surf):
    if (pol == 'h') and (surf == 'ice'):
        return R_hi(f)

    elif (pol == 'v') and (surf == 'ice'):
        return R_vi(f)

    elif (pol == 'h') and (surf == 'water'):
        return R_hw(f)

    elif (pol == 'v') and (surf == 'water'):
        return R_vw(f)

    else:
        return ValueError('invalid input arguments')

def get_emissivity(f,pol,surf):
    E = 1-get_reflectivity(f,pol,surf)
    return E

## Task 2

def calc_brightness_temp(f,pol,surf,T):
    T_b = get_emissivity(f,pol,surf)*T
    return T_b

#Do test:
print(calc_brightness_temp(50,'h','water',273))

#Do plot
frequency = np.linspace(1,100)
plt.plot(frequency,calc_brightness_temp(frequency,'h','water',273))
plt.plot(frequency,calc_brightness_temp(frequency,'v','water',273))
plt.legend(['horisontal','vertical'])
plt.xlabel('Frequency (GHz)')
plt.ylabel('Brightness Temp. (K)')
plt.title('Brightness temperature of water at 273 K')
plt.show()

## Task 3

def seaice_fm(f,pol,temp_ice,icefrac):
    if len(f) != len(pol):
        raise ValueError('numbers of channels in frequency different from polarisation')
    Tb = np.zeros(len(f))
    Temp_water = 273
    for i in range(len(f)):
        Tb[i] = calc_brightness_temp(f[i],pol[i],'water',Temp_water)*(1-icefrac) + calc_brightness_temp(f[i],pol[i],'ice',temp_ice)*icefrac
    
    return Tb

#Test case

print(seaice_fm([50],'h',270,0.7)[0])

## Task 5

def seaice_fm_noise(f,pol,temp_ice,icefrac,noise=0):
    if len(f) != len(pol):
        raise ValueError('numbers of channels in frequency different from polarisation')
    Tb = np.zeros(len(f))
    Temp_water = 273
    for i in range(len(f)):
        Tb[i] = calc_brightness_temp(f[i],pol[i],'water',Temp_water)*(1-icefrac) + calc_brightness_temp(f[i],pol[i],'ice',temp_ice)*icefrac
        Tb[i] = Tb[i] +  np.random.normal(0,noise)

    return Tb

#check that values is different from non_perturbed value
print(seaice_fm_noise([50],'h',270,0.7,10)[0])

##Task 6

def generate_jacobian(f,pol):
    Temp_water = 273
    K = np.matrix(np.zeros([len(f),2]))
    for i in range(len(f)):
        K[i,0] = -calc_brightness_temp(f[i],pol[i],'water',Temp_water)
        K[i,1] = get_emissivity(f[i],pol[i],'ice')

    return K 

def generate_y(Tb,pol,f):
    if len(Tb) != len(pol) or len(Tb) != len(f):
        raise ValueError('number of channels are inconsistent')
    Temp_water = 273
    y = np.matrix(np.zeros([len(Tb),1]))
    for i in range(len(Tb)):
        y[i,0] = Tb[i] - calc_brightness_temp(f[i],pol[i],'water',Temp_water)

    return y 

def inverse_model(Tb,f,pol):
    K = generate_jacobian(f,pol) #create jacobian
    y = generate_y(Tb,pol,f) #create measurement vector 

    x_hat = (np.linalg.inv(K.T.dot(K)).dot(K.T)).dot(y)
    return x_hat


f = np.array([19.7, 19.7, 37, 37, 85.5, 85.5])
pol = np.array(['h', 'v', 'h', 'v', 'h', 'v'])

temp_ice = 270
icefrac = 0.7
Tb = seaice_fm(f,pol,temp_ice,icefrac)
Tb_noise = seaice_fm_noise(f,pol,temp_ice,icefrac,5)

plt.plot(Tb)
plt.xlabel('Channel')
plt.ylabel('Brightness temperature (K)')
plt.title('Brightness temperature for icefrac 0.7 and temp_ice 270 K')
plt.show()

y_noise = generate_y(Tb_noise,pol,f)
x_true = np.matrix([icefrac,icefrac*temp_ice]).T #true state

#Check consistency of model
K = generate_jacobian(f,pol) #create jacobian
y = generate_y(Tb,pol,f) #create measurement vector 
y_true = K.dot(x_true) #check linearizatoin
print(y-y_true) #should be 0 if K is correctly calculated

x_hat = inverse_model(Tb,f,pol) #retrieval without noise
x_hat_noise = inverse_model(Tb_noise,f,pol) #retrieval with noise
print(x_hat-x_true)
print(x_hat_noise-x_true)

##Adding cloud layer

def seaice_fm_noise_cloud(f,pol,temp_ice,icefrac,lwp=0,T_cloud=283,phi=0,noise=0):
    
    Tb_ground = seaice_fm_noise(f,pol,temp_ice,icefrac,noise)
    t = 10**(-lwp*0.6*f**1.9/(10*np.cos(phi)))
    Tb_cloud = Tb_ground*t + (1-t)*T_cloud #Schwarchild equation

    return Tb_cloud

Tb_cloud = seaice_fm_noise_cloud(f,pol,temp_ice,icefrac,lwp=0.01)
x_hat_cloud = inverse_model(Tb_cloud,f,pol) #retrieval with cloud
print(x_hat_cloud-x_true)

