# Filtro de Kalman Extendido: Exemplo em Extensometria

# Neste exemplo, a medida de resistencia em um extensometro é utilizada
# para o calculo do raio de curvatura deste instrumento. O filtro de Kalman
# deve ser extendido, devido a nao-linearidade do problema.

import kalman
import numpy as np
import strainGauge as sg
import matplotlib.pyplot as plt


# Parametros do Modelo
c = 0.2            # Meia-altura maxima do extensometro, em cm
t0 = 0             # Tempo inicial
tf = 100           # Tempo final
R0 = 200           # Resistencia inicial, em ohm, sem extensao
rho0 = 8*R0*c      # Raio de curvatura inicial - 320 cm
n = 1001           # Pontos para discretizacao

t = np.linspace(t0, tf, n, endpoint=True)    # Vetor de tempos
dt = t[2] - t[1]                               # Passo de Tempo

## Simulação do Observador

# A variável que o simulador capta é a resistência R
#Res = 500                                  # Resistência Simulada
#y_r  = np.linspace(Res, Res, n)
#cov_ym = 45*100*c**2
#y_m = y_r + np.random.randn(len(y_r))*np.sqrt(cov_ym)

Res = sg.StrainGauge()
Res.rho = 160

def simFunction(x):
    y = x
    return y

Res.simFunction = simFunction
Res.simulate_array(n)        # Offline
y_r = Res.arrayPerfect
y_m = Res.array
cov_ym = Res.var


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
x0 = np.array([5])
P0 = np.array([1])
Fk = np.array([1])
R = np.array([10])
Q = np.array([10])

Filter = kalman.ExtendedKalmanFilter(x0, P0, Fk, R, Q)
Filter.h = h
Filter.HK = HK
Filter.filter(y_m)

radius_theoretical = ((8*R0*c)/(Res.perfect()-R0))
radius_theoretical = np.ones(n)*radius_theoretical
radius_filter = Filter.signal


## Exibicao

plt.figure(1)
plt.subplot(211)
plt.plot(t,y_r,'k', label = 'R - Modelo')
plt.plot(t,y_m,'r.', label = 'R- Observação')
plt.title('Resistência do Extensômetro')
plt.legend()

plt.subplot(2,1,2)
plt.plot(t, radius_filter,'m-', label = 'Raio de Curvatura - Filtro')
plt.plot(t, radius_theoretical, label = 'Raio de Curvatura - Teórico');
plt.title('Raio de Curvatura')
plt.legend()

plt.show()