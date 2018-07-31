# Filtro de Kalman Extendido: Exemplo em Extensometria

# Neste exemplo, a medida de resistencia em um extensometro é utilizada
# para o calculo do raio de curvatura deste instrumento. O filtro de Kalman
# deve ser extendido, devido a nao-linearidade do problema.

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
err.uniformLowHigh(-5,5)
sgTrue = ig.StrainGauge(Rzero, c, rho, Gf, err = err, n = n)

### Approximate Strain Gauge Model
Rzero         = 200    # Initial resistance
c             = 0.2    # Strain gauge half length (mm)
Gf            = 8      # Gauge factor
sgApproximate = ig.StrainGauge(Rzero, c, rho, Gf, err = 0, n = n)

### Approximation Error Random Variable
err = ig.ApproximationError(sgApproximate,sgTrue)

############################
###  SIMULATION ANALYSIS ###
############################

Rzero = 220
c = 0.2
Gf = 8.1
#rho = 160

n = 1000
strainGauge = ig.StrainGauge(Rzero, c, rho, Gf, err = err, n = n)
strainGaugePerfect = ig.StrainGauge(Rzero, c, rho, Gf, err = 0, n = n)

y_r = strainGaugePerfect.array
y_m = strainGauge.array
cov_ym = strainGauge.array.var()

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

radius_theoretical = strainGauge.stateArray
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