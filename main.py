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

n = 4000                                                        # Number of samples to test
rho = ig.RandomVariable(mean = 160, std = 5, dist = 'gaussian') # Radius of curvature

### True Strain Gauge Model
Rzero     = ig.RandomVariable(200, 10, 'gaussian')    # Initial resistance
c         = ig.RandomVariable(0.2, 0, 'nonRandom')    # Strain gauge half length (mm)
Gf        = ig.RandomVariable(8, 1, 'gaussian')       # Gauge factor
err       = ig.RandomVariable(0, 0,'uniform')
err.uniformLowHigh(-0.05, 0.05)
sgTrue = ig.StrainGauge(Rzero = Rzero, c = c, rho = rho, Gf = Gf, err = err, n = n)

### Approximate Strain Gauge Model
Rzero         = 200    # Initial resistance
c             = 0.2    # Strain gauge half length (mm)
Gf            = 8      # Gauge factor
sgApproximate = ig.StrainGauge(Rzero = Rzero, c = c, rho = rho, Gf = Gf, err = 0, n = n)

### Approximation Error Random Variable
err = ig.ApproximationError(sgApproximate,sgTrue)

############################
###  SIMULATION ANALYSIS ###
############################

n = 200

Rzero = 220
c = 0.2
Gf = 8.1
rho = 160
strainGaugeReal = ig.StrainGauge(Rzero, c, rho, Gf, err = err, n = n)

Rzero = 200
c = 0.2
Gf = 8
rho = 160
strainGaugeApproximate = ig.StrainGauge(Rzero, c, rho, Gf, err = 0, n = n)

y_r = strainGaugeApproximate.array
y_m = strainGaugeReal.array
cov_ym = strainGaugeReal.array.var()

def simFunction(x):
    #y = np.sin(0.25*x)
    #y = 1.0001*x
    #y = random.gauss(0,2)*x
    y = 1
    return y

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
x0 = np.array([10])
P0 = np.array([1])
Fk = np.array([0])
R = np.array([cov_ym])
Q = np.array([900])

Filter = kalman.ExtendedKalmanFilter(x0, P0, Fk, R, Q)
Filter.f = f
Filter.h = h
Filter.HK = HK
Filter.filter(y_m)

radius_theoretical = strainGaugeApproximate.stateArray
radius_filter = Filter.signal

## Plotting
plt.figure(1)
plt.subplot(211)
plt.plot(y_r,'k', label = 'R - Model')
plt.plot(y_m,'r.', label = 'R - Observation')
plt.title('Strain Gauge Resistance')
plt.legend()

plt.subplot(212)
plt.plot(radius_filter,'m-', label = 'Filtered')
plt.plot(radius_theoretical, label = 'True')
plt.title('Radius of Curvature')
plt.legend()

plt.show()
