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
from matplotlib.animation import FuncAnimation, writers
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
    

def ellipse_animation(ellipses, filtered_radius, path = None):
    
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set(xlim=(-160, 160), ylim=(-160, 160))

        t = np.linspace(0, 2*np.pi, 100)
        a, b = ellipses[0]['semiaxis']
        x, y = [a*np.cos(t), b*np.sin(t)]
        line = ax.plot(x, y, lw=2)[0]
        line.set_color('blue')
        line.set_linestyle('-')

        x,y = transpose(array(ellipses[0]['coordinates']))
        scat = ax.scatter(x, y, s=100)
        scat.set_color('white')
        scat.set_edgecolor('red')
        
        theta = np.arctan2(y,x)
        rho = array(ellipses[0]['radius_of_curvature'])
        u,v = [rho*np.cos(theta)*-1, rho*np.sin(theta)*-1]
        u,v = [np.round(u,2) + 0, np.round(v,2) + 0]
        arrows = ax.quiver(x, y, u, v)
        arrows.set_color('orange')
        arrows.set_label('True Radius of Curvature')
        
        u,v = [filtered_radius[0]*np.cos(theta)*-1, filtered_radius[0]*np.sin(theta)*-1]
        u,v = [np.round(u,2) + 0, np.round(v,2) + 0]
        arrows2 = ax.quiver(x, y, u, v, width = 0.005)
        arrows2.set_color('cyan')
        arrows2.set_label('Filtered Radius of Curvature')
        ax.legend(loc = 1)

        def update(i, fig, line, scat, arrows, arrows2):
        
                a, b = ellipses[i]['semiaxis']
                x, y = [a*np.cos(t), b*np.sin(t)]
                line.set_data(x, y)

                coord = array(ellipses[i]['coordinates'])
                scat.set_offsets(coord)

                x, y = transpose(array(ellipses[i]['coordinates']))
                theta = np.arctan2(y,x)
                rho = array(ellipses[i]['radius_of_curvature'])
                u,v = [rho*np.cos(theta)*-1, rho*np.sin(theta)*-1]
                u,v = [np.round(u,2) + 0, np.round(v,2) + 0]
                arrows.set_offsets(coord)
                arrows.set_UVC(u, v)

                rho = filtered_radius[i]
                u,v = [rho*np.cos(theta)*-1, rho*np.sin(theta)*-1]
                u,v = [np.round(u,2) + 0, np.round(v,2) + 0]
                arrows2.set_offsets(coord)
                arrows2.set_UVC(u, v)

                return line, scat, arrows, arrows2,

        anim = FuncAnimation(fig, update, fargs = (fig, line, scat, arrows, arrows2), frames=len(ellipses)-1, interval=50, blit=True)
        if path:
                anim.save(path + 'ellipse_animation.mp4', writer = 'ffmpeg')
        else:
                plt.draw()
                plt.show()


# def ellipse_animation2(ellipses, radius_filtered):  
    
#     t = np.linspace(0, 2*np.pi, 100)
#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1)
#     ax = plt.axes(xlim=(-160, 160), ylim=(-160, 160))

#     scat = ax.scatter([],[], s=100)
#     scat.set_color('white')
#     scat.set_edgecolor('red')

#     line = ax.plot([], [], lw=2)[0]
#     line.set_data([], [])
#     line.set_color('blue')
#     line.set_linestyle('-')

#     arrows = ax.quiver([], [])
#     arrows.set_color('orange')
#     arrows.set_label('True Radius of Curvature')

#     filteredArrows = ax.quiver([], [])
#     filteredArrows.set_color('magenta')
#     filteredArrows.set_label('Filtered Radius of Curvature')

#     ax.legend()

#     def update(i, fig, line, scat, arrows, filteredArrows):
        
#         a, b = ellipses[i]['semiaxis']
#         x, y = [a*np.cos(t), b*np.sin(t)]
#         line.set_data(x, y)

#         coord = array(ellipses[i]['coordinates'])
#         scat.set_offsets(coord)

#         x, y = np.transpose(np.array(ellipses[i]['coordinates']))
#         theta = np.arctan2(y,x)
#         rho = np.array(ellipses[i]['radius_of_curvature'])
#         u,v = [rho*np.cos(theta)*-1, rho*np.sin(theta)*-1]
#         arrows.set_offsets(coord)
#         arrows.set_UVC(u, v)

#         filteredArrows.set_offsets(coord)
#         u,v = [radius_filtered[i]*np.cos(theta)*-1, radius_filtered[i]*np.sin(theta)*-1]
#         filteredArrows.set_UVC(u, v)

#         return line, scat, arrows, filteredArrows,

#     anim = FuncAnimation(fig, update, fargs = (fig, line, scat, arrows, filteredArrows), frames=20, interval=50, blit=True)
#     #anim.save('ellipse_animation.mp4', writer='magick')    
#     plt.show()


# Modeling
trueData = json.load(open("ellipses.json","r"))
nSamples = trueData.__len__()
nGauges = trueData[0]['radius_of_curvature'].__len__()
#ellipse_animation(trueData)


### Function Models - Storage Retrieval

model_required = 6
models = json.load(open("models.json","r"))
for model in models:
    if model["id"] == model_required:
        break

data = []
for i in range(0, nSamples):
        
    rho = ig.RandomVariable()  
    rho.distributionArray = array(trueData[i]['radius_of_curvature'])    
    
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
    xTrue = sGT.measuredStateArray
    sGTwn   = ig.StrainGauge(Rzero, c, rho, Gf, err = noise)
    yMeasured = sGTwn.realizationArray

    data.append({'xTrue' : xTrue, 'yMeasured' : yMeasured})


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
Fk = array(model["filter_parameters"]["transition_matrix"])
R = np.eye(12)*approxErr.var
Q = array(model["filter_parameters"]["process_covariance"])

Filter = kalman.ExtendedKalmanFilter(x0, P0, Fk, R, Q)
Filter.f = f
Filter.h = h
Filter.H = H

Filtered    = kalman.Result(Filter, [k['yMeasured'] for k in data])
for i in range(0, nSamples):
    data[i]['xEstimated'] = xMeasured = Filtered.x[i]
    data[i]['yEstimated']  = h(Filtered.x[i]) - approxErr.mean

xEstimated  = array(Filtered.x)
yEstimated  = h(xEstimated) - approxErr.mean
covariance_trace = [k.trace() for k in Filtered.P]

###################
### ODR Fitting ###
###################

import scipy.odr

def Rcurvature(B, x):
        a = B[0]
        b = B[1]
        Rc = ((a*np.sin(x))**2+(b*np.cos(x))**2)**(3/2)
        Rc = Rc/(a*b)
        return Rc

fitModel = scipy.odr.Model(Rcurvature)

x = np.linspace(0, 2*np.pi, nGauges)
sx = np.ones(nGauges)*0.28

fitted_parameters = np.zeros([nSamples,2])
for i in range(nSamples):
        y = xEstimated[i]
        sy = Filtered.P[i].diagonal()
        data = scipy.odr.RealData(x, y, sx = sx, sy = sy)
        odr = scipy.odr.ODR(data, fitModel, beta0 = [144, 126])
        out = odr.run()
        fitted_parameters[i] = out.beta

################
### PLOTTING ###
################

imgDirectory = './media/' + model['name'] + '/'
ellipse_animation(trueData, xEstimated, imgDirectory)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.figure(0)
plt.subplot(111)
plt.plot(covariance_trace, label = 'Trace', c = 'g')
plt.title('Trace of Covariance Matrix')
plt.savefig(imgDirectory + "torax_expanding_covariance_trace.png", dpi=96)

for i in range(0, nGauges):

    plt.figure(i+1, figsize = (10,6))
    plt.subplot(211)
    plt.plot(array([k['yMeasured'] for k in data])[:,i], label = 'Measurement: R' + str(i))
    plt.plot(array([k['yEstimated'] for k in data])[:,i], label = 'Estimation: R' + str(i))
    plt.ylabel('Resistance (' + r'$\Omega$' + ')')
    plt.title('Strain Gauge Resistance')
    plt.legend()

    plt.subplot(212)
    plt.plot(array([k['xEstimated'] for k in data])[:,i],'g-', label = 'Estimation: R' + str(i))
    plt.plot(array([k['xTrue'] for k in data])[:,i], label = 'True: ' + r'$\rho$' + str(i))
    plt.ylim(0, data[i]['xTrue'].max()*1.5)
    plt.ylabel('Radius of Curvature (' + r'$\rho$' + ')')
    plt.title('Radius of Curvature')
    plt.legend()
    
    plt.savefig(imgDirectory + "torax_expanding_" + str(i) + ".png", dpi=96)

print("Program finished.")
