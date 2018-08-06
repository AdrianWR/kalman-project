# -*- coding: utf-8 -*-
# Extended Kalman Filter: Strain Gauge Example

# In this example, a strain gauge resistance measurement is used to
# calculate its radius of curvature. The Kalman Filter must be of the
# extended type, regarding the model non-linearity.

import kalman
import random
import numpy as np
import InstrumentGenerator as ig
import matplotlib.pyplot as plt

#############################################
### APPROXIMATION ERROR METHOD SIMULATION ###
#############################################

n = 4000                                                               # Number of samples to test
rho = ig.RandomVariable(mean = 160, std = 5, dist = 'gaussian', n = n) # Radius of curvature

### Approximate Strain Gauge Model
Rzero         = 200    # Initial resistance
c             = 0.2    # Strain gauge half length (mm)
Gf            = 8      # Gauge factor
sgApproximate = ig.StrainGauge(Rzero, c, rho, Gf)

### Real Strain Gauge Model
Rzero     = ig.RandomVariable(200, 10, 'gaussian')    # Initial resistance
c         = ig.RandomVariable(0.2, 0, 'nonRandom')    # Strain gauge half length (mm)
Gf        = ig.RandomVariable(8, 1, 'gaussian')       # Gauge factor
err       = ig.RandomVariable(0, 0,'uniform')
err.uniformLowHigh(-5, 5)
sgReal = ig.StrainGauge(Rzero, c, rho, Gf, err = err)

### Approximation Error Random Variable
err = ig.ApproximationError(sgApproximate,sgReal)

############################
###  SIMULATION ANALYSIS ###
############################

def simFunction(x):
    #y = 200*np.sin(0.2*x)
    #y = 1.0001*x
    #y = random.gauss(0,2)*x
    y = np.full(x.shape, 200)
    return y

n = 200
rho = ig.RandomVariable(0, 0, 'nonRandom', n)
rho.distributionArray = simFunction(np.array(range(1,n+1)))

# Strain Gauge Approximate Model for Analysis
Rzero = 200
c = 0.2
Gf = 8
strainGaugeApproximate = ig.StrainGauge(Rzero, c, rho, Gf, err = 0)

# Strain Gauge Real Model for Analysis
Rzero = 217
c = 0.18
Gf = 8.1
err     = ig.RandomVariable(dist = 'uniform')
err.uniformLowHigh(-1, 1)
strainGaugeReal = ig.StrainGauge(Rzero, c, rho, Gf, err = err)

y_a = strainGaugeApproximate.realizationArray
y_m = strainGaugeReal.realizationArray
cov_ym = strainGaugeReal.realizationArray.var()



########################
### KALMAN FILTERING ###
########################

### Strain Gauge Process Equations

# Process Function
### Random Walk Model
def f(x):
    return x

# Observer Function
def h(x):  
    R = ((Gf*Rzero*c)/x) + Rzero + err.mean
    return R

# Observer Derivative Function
def HK(x):
    dR = -(Gf*Rzero*c)/(x**2)
    return dR

# Filter Parameters
x0 = np.array([100])
P0 = np.array([1])
Fk = np.array([1])
R = np.array([cov_ym])
Q = np.array([10])

Filter = kalman.ExtendedKalmanFilter(x0, P0, Fk, R, Q)
Filter.f = f
Filter.h = h
Filter.HK = HK
Filter.filterSampleArray(y_m)

radius_measurement = strainGaugeReal.stateFromRealization()
radius_approximate = strainGaugeReal.rho.distributionArray
radius_filter = np.array(Filter.signal)

################
### PLOTTING ###
################

plt.figure(num = 1, figsize=(7,8))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.tight_layout()

plt.subplot(211)
plt.plot(y_a,'k', label = 'R - Approximation')
plt.plot(y_m,'r.', label = 'R - Observation')
plt.ylim(y_a.min()*0.95, y_m.max()*1.05)
plt.ylabel('Resistance (' + r'$\Omega$' + ')')
plt.title('Strain Gauge Resistance')
plt.legend()

plt.subplot(212)
plt.plot(radius_measurement, 'g--', label = 'Measured')
plt.plot(radius_filter,'m-', label = 'Filtered')
plt.plot(radius_approximate, label = 'Real')
plt.ylabel('Radius of Curvature (' + r'$\rho$' + ')')
plt.title('Radius of Curvature')
plt.legend()

plt.savefig("./simulation.png",dpi=72)
plt.show()
print('ok')