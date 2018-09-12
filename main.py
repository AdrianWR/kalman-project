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
from os.path import exists

#############################################
### APPROXIMATION ERROR METHOD SIMULATION ###
#############################################



def callApproximationError():

    n = 4000                                                               # Number of samples to test
    rho = ig.RandomVariable(mean = 190, std = 20, dist = 'gaussian', n = n) # Radius of curvature

    ### Approximate Strain Gauge Model
    Rzero         = 9000    # Initial resistance
    c             = 1.7    # Strain gauge half length (mm)
    Gf            = 8      # Gauge factor
    sgApproximate = ig.StrainGauge(Rzero, c, rho, Gf)

    ### Real Strain Gauge Model
    Rzero     = ig.RandomVariable(9000, 100, 'gaussian')    # Initial resistance
    c         = ig.RandomVariable(1.7, 0, 'nonRandom')    # Strain gauge half length (mm)
    Gf        = ig.RandomVariable(8, 1, 'gaussian')       # Gauge factor
    err       = ig.RandomVariable(0, 0,'uniform')
    err.uniformLowHigh(-0.5, 0.5)
    sgReal = ig.StrainGauge(Rzero, c, rho, Gf, err = err)

    ### Approximation Error Random Variable
    return ig.ApproximationError(sgApproximate,sgReal)

Filename = 'ApproximationError.json'
if exists(Filename):
    with open(Filename, "r") as read_file:
        data = json.load(read_file)
    err = ig.ApproximationError()
    err.mean = data['mean']
    err.std = data['std']
    err.dist = data['dist']
else:
    err = callApproximationError()
    with open(Filename, "w") as write_file:
        json.dump(err.__dict__, write_file)

############################
###  SIMULATION ANALYSIS ###
############################

def simFunction(x):
    #y = 200*np.sin(0.01*x) + 100 # Sine Function
    #y = np.full(x.shape, 10)      # Constant Function
    y = -0.25*x + 100              # Linear Function
    return y

n = 300
rho = ig.RandomVariable(0, 0, 'nonRandom', n)
rho.distributionArray = simFunction(np.array(range(1,n+1)))

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
cov_yMeasured = err.var

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

def h_noError(x):
    return ((Gf*Rzero*c)/x) + Rzero

# Observer Derivative Function
def H(x):
    dR = -(Gf*Rzero*c)/(x**2)
    return dR

# Filter Parameters
x0 = np.array([15])
P0 = np.array([10**2])
Fk = np.array([1]) #transition matrix
R = np.array([cov_yMeasured])
Q = np.array([1**2])

Filter = kalman.ExtendedKalmanFilter(x0, P0, Fk, R, Q)
Filter.f = f
Filter.h = h
Filter.H = H

radius_true     = strainGaugeTrue.measuredStateArray
radius_filter   = Filter.filterSampleArray(yMeasured)

yEstimated = h_noError(radius_filter)

################
### PLOTTING ###
################

plt.figure(num = 1, figsize=(14,8))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
#plt.tight_layout()

plt.subplot(211)
plt.plot(yTrue,'k', label = 'R - True')
plt.plot(yMeasured,'r.', label = 'R - Measurements')
plt.plot(yEstimated, 'b-', label = 'R - Estimated')
plt.ylim(yTrue.min()*0.95, yTrue.max()*1.05)
plt.ylabel('Resistance (' + r'$\Omega$' + ')')
plt.title('Strain Gauge Resistance')
plt.legend()

plt.subplot(212)
#plt.plot(radius_measurement, 'g--', label = 'Measured')
plt.plot(radius_filter,'g-', label = 'Estimated')
plt.plot(radius_true, label = 'True')
plt.ylabel('Radius of Curvature (' + r'$\rho$' + ')')
plt.title('Radius of Curvature')
plt.legend()
#plt.yscale('log')

plt.savefig("./simulation_linear.png", metadata = {'x0': str(x0), 'P0': str(P0), 'R': str(R), 'Q': str(Q)}, dpi=96)
plt.show()
#print('ok')