# -*- coding: utf-8 -*-
# Extended Kalman Filter: Strain Gauge Example

# In this example, a strain gauge resistance measurement is used to
# calculate its radius of curvature. The Kalman Filter must be of the
# extended type, regarding the model non-linearity.

from subprocess import check_output
import InstrumentGenerator as ig
import matplotlib.pyplot as plt
import numpy as np
import kalman
import json

#############################################
### APPROXIMATION ERROR METHOD SIMULATION ###
#############################################

approxErr = ig.ApproximationError()
approxErr.simulateApproximationError(4000, re_simulate = False)

############################
###  SIMULATION ANALYSIS ###
############################

# Modeling

chestData = check_output(["Rscript","bodydata.r"], shell = True)
chestData = chestData.decode("UTF-8")
chestData = json.loads(chestData.splitlines()[0])
#chestData = json.loads
rc_min = chestData[0]
rc_max = chestData[1]

### Function Models - Storage Retrieval

# CHANGE HERE!!!
model_required = 4
models = json.load(open("models.json","r"))
for model in models:
    if model["id"] == model_required:
        break


### Computational Parameters

n = 200
t = np.array(range(1,n+1))
def y_min(t): return np.full(t.shape, rc_min["mean"])
def y_max(t): return np.full(t.shape, rc_max["mean"])


#rho = []
strainGaugeTrueWithNoise = []
strainGaugeApproximate = []
yMeasured = []
for i in range(0,4):
    #gauge = ig.RandomVariable(rc_max["mean"], rc_max["std"], 'gaussian', n)
    
    rho = ig.RandomVariable()
    if (i % 2 == 0):        
        rho.distributionArray = y_max(t)
    else:
        rho.distributionArray = y_min(t)
    
    Rzero = 9000
    c = 1.7
    Gf = 8
    sGA = ig.StrainGauge(Rzero, c, rho, Gf, err = 0)
    strainGaugeApproximate.append(sGA)

    Rzero   = ig.RandomVariable(Rzero, 100, 'gaussian')
    c       = ig.RandomVariable(c, 0.1, 'gaussian')
    Gf      = ig.RandomVariable(Gf, 0.4, 'gaussian')
    noise = ig.RandomVariable(dist = 'uniform')
    noise.uniformLowHigh(-500, 500)
    sGTwn   = ig.StrainGauge(Rzero(), c(), rho, Gf(), err = noise)
    strainGaugeTrueWithNoise.append(sGTwn)
    yMeasured.append(sGTwn.realizationArray)

yMeasured = np.array(yMeasured)
yMeasured = np.transpose(yMeasured)




#rho1 = ig.RandomVariable(0, 0, 'nonRandom', n)
#rho.distributionArray = y(t)

# ----------------------- #
# - Strain Gauge Models - #
# ----------------------- #

# Strain Gauge True Model
# Rzero = 9216
# c = 1.532
# Gf = 8.1
# strainGaugeTrue = ig.StrainGauge(Rzero, c, rho, Gf, err = 0)

# # Strain Gauge True Model With Noise
# noise = ig.RandomVariable(dist = 'uniform')
# noise.uniformLowHigh(-500, 500)
# strainGaugeMeasured = ig.StrainGauge(Rzero, c, rho, Gf, err = noise)

# # Strain Gauge Approximate Model
# Rzero = 9000
# c = 1.7
# Gf = 8
# strainGaugeApproximate = ig.StrainGauge(Rzero, c, rho, Gf, err = 0)

#yTrue = strainGaugeTrue.realizationArray
#yMeasured = strainGaugeMeasured.realizationArray
### cov_yMeasured = approxErr.var

########################
### KALMAN FILTERING ###
########################

### Strain Gauge Process Equations

Rzero = 9000
c = 1.7
Gf = 8

# Process FunctionyMeasured
### Random Walk Model
def f(x):
    return x

# Observer Function
def h(x):  
    R = ((Gf*Rzero*c)/x) + Rzero + approxErr.mean
    return R

# Observer Derivative Function
def H(x):
    dR = -(Gf*Rzero*c)/(x**2)
    return dR

# Filter Parameters
x0 = np.array(model["filter_parameters"]["initial_x"])
P0 = np.array(model["filter_parameters"]["initial_p"])
Fk = np.eye(4)                                          #transition matrix
R = np.ones(4)*approxErr.var
Q = np.array(model["filter_parameters"]["process_covariance"])

Filter = kalman.ExtendedKalmanFilter(x0, P0, Fk, R, Q)
Filter.f = f
Filter.h = h
Filter.H = H

Filtered         = kalman.Result(Filter, yMeasured)
#radius_true      = strainGaugeTrue.measuredStateArray
radius_filter    = Filtered.x
error_covariance = Filtered.P
#yEstimated       = h(radius_filter) - approxErr.mean

################
### PLOTTING ###
################

plt.figure(1)#num = 1, figsize=(8,8))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
#plt.tight_layout()

#plt.subplot(211)
#plt.plot(yTrue,'k', label = 'R - True')
#plt.plot(yMeasured, label = 'Measurement - R')
for i in range(0,4):
    #plt.plot(yEstimated[:,i], label = 'R - Estimated' + str*(i))
    plt.plot(yMeasured[:,i], label = 'Measurement: R' + str(i))
plt.ylim(yMeasured.min()*0.95, yMeasured.max()*1.05)
plt.ylabel('Resistance (' + r'$\Omega$' + ')')
plt.title('Strain Gauge Resistance')
plt.legend()
plt.show()

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