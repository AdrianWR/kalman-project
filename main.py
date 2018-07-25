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
rho = ig.RandomVariable(mean = 320, std = 5, dist = 'gaussian') # Radius of curvature

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

n = 2001
strainGauge = ig.StrainGauge(Rzero, c, rho, Gf, err = err, n = n)
strainGaugePerfect = ig.StrainGauge(Rzero, c, rho.mean, Gf, n = n)

# t0 = 0             # Tempo inicial
# tf = 100           # Tempo final
# t = np.linspace(t0, tf, n, endpoint=True)    # Vetor de tempos
# dt = t[2] - t[1]                               # Passo de Tempo

## Simulação do Observador

# A variável que o simulador capta é a resistência R
#Res = 500                                  # Resistência Simulada
#y_r  = np.linspace(Res, Res, n)
#cov_ym = 45*100*c**2
#y_m = y_r + np.random.randn(len(y_r))*np.sqrt(cov_ym)

def simFunction(x):
    #y = np.sin(0.25*x)
    #y = 1.0001*x
    #y = random.gauss(0,2)*x
    y = 1
    return y

y_r = strainGaugePerfect.array
y_m = strainGauge.array
cov_ym = strainGauge.array.var()

########################
### KALMAN FILTERING ###
########################

### Strain Gauge Process Equations

# Função do Observador
def h(x):  
    R = ((Gf*Rzero*c)/x) + Rzero + err.mean
    return R

# Diferenciação do Observador
def HK(x):
    dR = -(Gf*Rzero*c)/(x**2)
    return dR

# Inicializar objeto de filtro
x0 = np.array([300])
P0 = np.array([1])
Fk = np.array([1])
R = np.array([cov_ym])
Q = np.array([1])

Filter = kalman.ExtendedKalmanFilter(x0, P0, Fk, R, Q)
Filter.h = h
Filter.HK = HK
Filter.filter(y_m)

#radius_theoretical = ((8*R0*c)/(Res.arrayPerfect-R0*np.ones(n)))
#radius_theoretical = np.ones(n)*radius_theoretical
radius_theoretical = strainGauge.stateArray
radius_filter = Filter.signal


## Exibicao

plt.figure(1)
plt.subplot(211)
plt.plot(y_r,'k', label = 'R - Modelo')
plt.plot(y_m,'r.', label = 'R- Observação')
plt.title('Resistência do Extensômetro')
plt.legend()

plt.subplot(212)
plt.plot(radius_filter,'m-', label = 'Raio de Curvatura - Filtro')
plt.plot(radius_theoretical, label = 'Raio de Curvatura - Teórico')
plt.title('Raio de Curvatura')
plt.legend()

plt.show()
