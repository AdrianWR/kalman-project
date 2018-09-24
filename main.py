# -*- coding: utf-8 -*-
# Extended Kalman Filter: Strain Gauge Example

# In this example, a strain gauge resistance measurement is used to
# calculate its radius of curvature. The Kalman Filter must be of the
# extended type, regarding the model non-linearity.

import json
import kalman
import numpy as np
import InstrumentGenerator as ig
import matplotlib.pyplot as plt
#from os.path import exists

#############################################
### APPROXIMATION ERROR METHOD SIMULATION ###
#############################################

approxErr = ig.ApproximationError()
approxErr.simulateApproximationError(4000, re_simulate = True)

############################
###  SIMULATION ANALYSIS ###
############################

### Functions Storage


def fsim(t): return t**3

    #y = 200*np.sin(0.01*x) + 100 # Sine Function
    #y = np.full(x.shape, 10)      # Constant Function
    #y = -0.25*x + 100              # Linear Function

n = 300
rho = ig.RandomVariable(0, 0, 'nonRandom', n)

t = np.array(range(1,n+1))
rho.distributionArray = fsim(t)

# Strain Gauge True Model for Analysis
Rzero = 9216
c = 1.532
Gf = 8.1
strainGaugeTrue = ig.StrainGauge(Rzero, c, rho, Gf, err = 0)

noise = ig.RandomVariable(dist = 'uniform')
noise.uniformLowHigh(-500, 500)
strainGaugeMeasured = ig.StrainGauge(Rzero, c, rho, Gf, err = noise)

# Strain Gauge Approximate Model for Analysis
Rzero = 9000
c = 1.7
Gf = 8
strainGaugeApproximate = ig.StrainGauge(Rzero, c, rho, Gf, err = 0)

yTrue = strainGaugeTrue.realizationArray
yMeasured = strainGaugeMeasured.realizationArray
cov_yMeasured = approxErr.var

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

#def h_noError(x):
#    return ((Gf*Rzero*c)/x) + Rzero

# Observer Derivative Function
def H(x):
    dR = -(Gf*Rzero*c)/(x**2)
    return dR

# Filter Parameters
x0 = np.array([15])
P0 = np.array([10**2])
Fk = np.array([1]) #transition matrix
R = np.array([approxErr.var])
Q = np.array([0.01**2])

Filter = kalman.ExtendedKalmanFilter(x0, P0, Fk, R, Q)
Filter.f = f
Filter.h = h
Filter.H = H

Filtered         = kalman.Result(Filter, yMeasured)
radius_true      = strainGaugeTrue.measuredStateArray
radius_filter    = Filtered.x
error_covariance = Filtered.P

yEstimated = h_noError(radius_filter)

################
### PLOTTING ###
################

plt.figure(1)#num = 1, figsize=(8,8))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
#plt.tight_layout()

#plt.subplot(211)
plt.plot(yTrue,'k', label = 'R - True')
plt.plot(yMeasured,'r.', label = 'R - Measurements')
plt.plot(yEstimated, 'b-', label = 'R - Estimated')
plt.ylim(yTrue.min()*0.95, yTrue.max()*1.05)
plt.ylabel('Resistance (' + r'$\Omega$' + ')')
plt.title('Strain Gauge Resistance')
plt.legend()
#plt.show()

plt.figure(2)
#plt.subplot(212)
#plt.plot(radius_measurement, 'g--', label = 'Measured')
plt.plot(radius_filter,'g-', label = 'Estimated')
plt.plot(radius_true, label = 'True')
plt.ylabel('Radius of Curvature (' + r'$\rho$' + ')')
plt.title('Radius of Curvature')
plt.legend()
#plt.yscale('log')

plt.figure(3)
# plt.subplot(313)
plt.plot(error_covariance, label = 'Error Covariance')
plt.ylabel('Radius of Curvature (' + r'$\rho$' + ')')
plt.title('Error Covariance')
plt.legend()

#plt.savefig("./simu.png", metadata = {'x0': str(x0), 'P0': str(P0), 'R': str(R), 'Q': str(Q)}, dpi=96)
plt.show()
#print('ok')