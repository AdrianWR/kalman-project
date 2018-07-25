# Filtro de Kalman Extendido: Exemplo em Extensometria

# Neste exemplo, a medida de resistencia em um extensometro é utilizada
# para o calculo do raio de curvatura deste instrumento. O filtro de Kalman
# deve ser extendido, devido a nao-linearidade do problema.

import kalman
import random
import numpy as np
import InstrumentGenerator as ig
import matplotlib.pyplot as plt

# Models
Rzero = ig.RandomVariable(200, 10, 'gaussian')    # initial resistance
c     = ig.RandomVariable(0.2)                    # strain gauge half length (mm)
rho   = ig.RandomVariable(320)                    # radius of curvature
Gf    = ig.RandomVariable(10, 1, 'gaussian')      # gauge factor

n = 201            # Number of samples

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

#Res = ig.StrainGauge()
#Res.rho = random.gauss(10,1)

def simFunction(x):
    #y = np.sin(0.25*x)
    y = 1
    #y = 1.0001*x
    #y = random.gauss(0,2)*x
    return y

Res.simFunction = simFunction
Res.simulateArray(n)        # Offline
y_r = Res.arrayPerfect
y_m = Res.array
cov_ym = Res.errorVar


####  Kalman

### Equações de Processo: Extensômetro

# O observador precisará tem um ruído adicionado
# epsilon, que é conhecido pelos testes.

# Função do Observador
def h(x):
    R = ((8*R0*c)/x) + R0
    return R

# Diferenciação do Observador
def HK(x):
    dR = -(8*R0*c)/(x**2)
    return dR

# Inicializar objeto de filtro
x0 = np.array([1])
P0 = np.array([1])
Fk = np.array([1])
R = np.array([10])
Q = np.array([0.01])

Filter = kalman.ExtendedKalmanFilter(x0, P0, Fk, R, Q)
Filter.h = h
Filter.HK = HK
Filter.filter(y_m)

#radius_theoretical = ((8*R0*c)/(Res.arrayPerfect-R0*np.ones(n)))
#radius_theoretical = np.ones(n)*radius_theoretical
radius_theoretical = Res.exactStateArray
radius_filter = Filter.signal


## Exibicao

plt.figure(1)
plt.subplot(211)
plt.plot(t,y_r,'k', label = 'R - Modelo')
plt.plot(t,y_m,'r.', label = 'R- Observação')
plt.title('Resistência do Extensômetro')
plt.legend()

plt.subplot(212)
plt.plot(t, radius_filter,'m-', label = 'Raio de Curvatura - Filtro')
plt.plot(t, radius_theoretical, label = 'Raio de Curvatura - Teórico')
plt.title('Raio de Curvatura')
plt.legend()

plt.show()
