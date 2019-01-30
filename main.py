# -*- coding: utf-8 -*-
# Extended Kalman Filter: Strain Gauge Example

# In this example, a strain gauge resistance measurement is used to
# calculate its radius of curvature. The Kalman Filter must be of the
# extended type, regarding the model non-linearity.

import InstrumentGenerator as ig
import matplotlib.pyplot as plt
import numpy as np
import kalman
import json
from numpy import array, transpose
from subprocess import check_output


#############################################
### APPROXIMATION ERROR METHOD SIMULATION ###
#############################################

approxErr = ig.ApproximationError()
approxErr.simulateApproximationError(4000, re_simulate = False)

############################
###  SIMULATION ANALYSIS ###
############################

### Function Models - Storage Retrieval

# CHANGE HERE!!!
model_required = 1
models = json.load(open("models.json","r"))
for model in models:
    if model["id"] == model_required:
        break

### Computational Parameters

n = 1000
t = np.array(range(1,n+1))
def y(t): return eval(model["function"])
rho = ig.RandomVariable(0, 0, 'nonRandom', n)
rho.distributionArray = y(t)

# ----------------------- #
# - Strain Gauge Models - #
# ----------------------- #

# Strain Gauge True Model
Rzero = 9216
c = 1.532
Gf = 8.1
strainGaugeTrue = ig.StrainGauge(Rzero, c, rho, Gf, err = 0)

# Strain Gauge True Model With Noise
noise = ig.RandomVariable(dist = 'uniform')
noise.uniformLowHigh(-500, 500)
strainGaugeMeasured = ig.StrainGauge(Rzero, c, rho, Gf, err = noise)

# Strain Gauge Approximate Model
Rzero = 9000
c = 1.7
Gf = 8
strainGaugeApproximate = ig.StrainGauge(Rzero, c, rho, Gf, err = 0)

yTrue       = transpose(array([strainGaugeTrue.realizationArray]))
yMeasured   = transpose(array([strainGaugeMeasured.realizationArray]))
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
    R = ((Gf*Rzero*c)/x) + Rzero + approxErr.mean
    return R

# Observer Derivative Function
def H(x):
    dR = -(Gf*Rzero*c)/(x**2)
    return dR

# Filter Parameters
x0 = array([model["filter_parameters"]["initial_x"]])
P0 = array([model["filter_parameters"]["initial_p"]])
Fk = array([[1]])                                          #transition matrix
R = array([[approxErr.var]])
Q = array([model["filter_parameters"]["process_covariance"]])

Filter = kalman.ExtendedKalmanFilter(x0, P0, Fk, R, Q)
Filter.f = f
Filter.h = h
Filter.H = H

Filtered            = kalman.Result(Filter, yMeasured)
xTrue               = transpose(array([strainGaugeTrue.measuredStateArray]))
xEstimated          = array(Filtered.x)
yEstimated          = (h(array(Filtered.x)) - approxErr.mean)
error_covariance    = array(Filtered.P)

################
### PLOTTING ###
################


imgDirectory = './media/' + model['name'] + '/'
for i in range(len(xTrue[0])):

    plt.figure(i, figsize = (10,6))
    plt.subplot(211)
    plt.plot(yMeasured[:,i], label = 'Medidas: R' + str(i))
    plt.plot(yEstimated[:,i], label = 'Estimativas: R' + str(i))
    #plt.ylim(yMeasured.min()*0.95, yMeasured.max()*1.05)
    plt.ylabel('Resistencia (' + r'$\Omega$' + ')')
    plt.title('Resistencia do Extensometro')
    plt.legend()

    plt.subplot(212)
    plt.plot(xEstimated[:,i],'g-', label = 'Estimativa: ' + r'$\rho$' + str(i))
    plt.plot(xTrue[:,i], label = 'Verdadeiro: ' + r'$\rho$' + str(i))
    plt.ylim(0, xTrue.max()*1.5)
    plt.ylabel('Raio de Curvatura (' + r'$\rho$' + ')')
    plt.title('Raio de Curvatura')
    plt.legend()

    #plt.show()
    plt.savefig(imgDirectory + "estimativa.png", dpi=96)

plt.figure(1, figsize = (10,3))
plt.plot(error_covariance[:,0], label = 'Covariancia do Erro')
plt.title('Covariancia do Estado')
plt.ylabel('Covariancia da Resistencia (' + r'$\Omega^2$' + ')')
#plt.legend()
plt.savefig(imgDirectory + "covariancia.png", dpi=96)