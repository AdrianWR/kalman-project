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
from matplotlib.animation import FuncAnimation, FFMpegWriter
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

def draw_ellipse(ellipse):
    
    a, b = ellipse['semiaxis']
    t = np.linspace(0, 2*np.pi, 100)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(*zip(*ellipse['coordinates']))
    ax.plot(a*np.cos(t), b*np.sin(t), 'g--')
    ax.set_xlim(-160, 160)
    ax.set_ylim(-160, 160)
    
    plt.show()
    

def ellipse_animation(ellipses):

    t = np.linspace(0, 2*np.pi, 100)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax = plt.axes(xlim=(-160, 160), ylim=(-160, 160))

    #coordinates = array(ellipses[0]['coordinates'])
    scat = ax.scatter([],[], s=100)
    scat.set_color('white')
    scat.set_edgecolor('red')

    line = ax.plot([], [], lw=2)[0]
    line.set_data([], [])
    line.set_color('blue')
    line.set_linestyle('-')

    def update(i, fig, line, scat):
        
        a, b = ellipses[i]['semiaxis']
        x, y = [a*np.cos(t), b*np.sin(t)]
        line.set_data(x, y)

        coord = array(ellipses[i]['coordinates'])
        scat.set_offsets(coord)
        
        return line, scat,

    anim = FuncAnimation(fig, update, fargs = (fig, line, scat), frames=20, interval=50, blit=True)
    plt.show()


# Modeling
data = json.load(open("ellipses.json","r"))
#ellipse_animation(data)
#draw_ellipse(data[0])
#draw_ellipse(data[-1])

### Function Models - Storage Retrieval

# CHANGE HERE!!!
model_required = 5
models = json.load(open("models.json","r"))
for model in models:
    if model["id"] == model_required:
        break


# class EllipseModel(object):

#     def __init__(self, nGauges):
#         self.nGauges = nGauges

### Computational Parameters

n = 100
t = array(range(1,n+1))
def y(t, const): return np.full(t.shape, const)




for i in range(0, data.__len__()):
    yMeasured = []
    xTrue = []
    for j in range(0, data[0]['radius_of_curvature'].__len__()):
        
        rho = ig.RandomVariable()  
        rho.distributionArray = y(t, data[0]['radius_of_curvature'][j])
        
        Rzero = 9000
        c = 1.7
        Gf = 8

        Rzero   = ig.RandomVariable(Rzero, 100, 'gaussian')
        Rzero   = Rzero()
        c       = ig.RandomVariable(c, 0.1, 'gaussian')
        c       = c()
        Gf      = ig.RandomVariable(Gf, 0.4, 'gaussian')
        Gf      = Gf()
        noise = ig.RandomVariable(dist = 'uniform')
        noise.uniformLowHigh(-200, 200)
        sGT     = ig.StrainGauge(Rzero, c, rho, Gf, err = 0)
        xTrue.append(sGT.measuredStateArray)
        sGTwn   = ig.StrainGauge(Rzero, c, rho, Gf, err = noise)
        yMeasured.append(sGTwn.realizationArray)

    yMeasured   = array(yMeasured)
    yMeasured   = transpose(yMeasured)
    xTrue       = array(xTrue)
    xTrue       = transpose(xTrue)

########################
### KALMAN FILTERING ###
########################

### Strain Gauge Process Equations

# Approximate Variables
Rzero = 9000
c = 1.7
Gf = 8

# Process Function

### Random Walk Model
def f(x):
    return x

### Observer Function
def h(x):  
    R = ((Gf*Rzero*c)/x) + Rzero + approxErr.mean
    return R

### Observer Derivative Function
def H(x):
    dR = -(Gf*Rzero*c)/(x**2)
    return dR

### Filter Parameters
x0 = array(model["filter_parameters"]["initial_x"])
P0 = array(model["filter_parameters"]["initial_p"])
#Fk = np.eye(4)
Fk = array(model["filter_parameters"]["transition_matrix"])        #transition matrix
R = np.eye(12)*approxErr.var.__round__(2)
Q = array(model["filter_parameters"]["process_covariance"])

Filter = kalman.ExtendedKalmanFilter(x0, P0, Fk, R, Q)
Filter.f = f
Filter.h = h
Filter.H = H

Filtered    = kalman.Result(Filter, yMeasured)
xEstimated  = array(Filtered.x)
yEstimated  = h(xEstimated) - approxErr.mean
#covarianceEigenvalues, v = np.linalg.eig(array(Filtered.P))
#covarianceEigenvalues = np.real(covarianceEigenvalues)


################
### PLOTTING ###
################

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
#plt.tight_layout()

for i in range(4):

    plt.figure(i+1, figsize = (10,6))
    plt.subplot(211)
    plt.plot(yMeasured[:,i], label = 'Measurement: R' + str(i))
    plt.plot(yEstimated[:,i], label = 'Estimation: R' + str(i))
    #plt.ylim(yMeasured.min()*0.95, yMeasured.max()*1.05)
    plt.ylabel('Resistance (' + r'$\Omega$' + ')')
    plt.title('Strain Gauge Resistance')
    plt.legend()

    plt.subplot(212)
    plt.plot(xEstimated[:,i],'g-', label = 'Estimation: R' + str(i))
    plt.plot(xTrue[:,i], label = 'True: ' + r'$\rho$' + str(i))
    plt.ylim(0, xTrue.max()*1.5)
    plt.ylabel('Radius of Curvature (' + r'$\rho$' + ')')
    plt.title('Radius of Curvature')
    plt.legend()
    
    # plt.figure(0)
    # plt.plot(covarianceEigenvalues[:,i], label = 'Error Covariance '+ r'$\rho$' + str(i))
    # plt.yscale('log')
    # plt.title('Error Covariance')
    # plt.legend()


    #plt.savefig("./images/torax_R" + str(i) + ".png", dpi=96)

plt.show()
print("Program finished.")

#P = array(Filtered.P)

