import numpy as np
import random

# Esta classe cria um objeto Extensometro, simulando parametros fisicos
# do instrumento. As constantes inicializaveis podem ser alteradas após
# a instanciação do objeto, sendo as principais Resistencia Inicial (Rzero),
# meia altura do extensometro (c) e raio de curvatura (rho).


#aDICIONAR erro de medida vk
class StrainGauge:        

    def __init__(self, Rzero = 200, c = 0.2):
        self.Rzero  = Rzero
        self.c      = c
        self.rho    = 8*Rzero*c
        pass

    def rho(self)

    # Função de teste simulando variação de resistência com o tempo.
    def simFunction(t):
        return 1
        
    # Função que retorna o valor da Resistencia levando os erros em conta
    def realData(self, rho):
        Gf = random.gauss(8,1)
        #Rzero = random.gauss(200, 10)
        Rzero = self.Rzero
        c = self.c
        R = Gf*Rzero*c/rho + Rzero
        return R

    # Função que retorna o valor da Resistencia sem erros associados
    def exactData(self, rho):
        Gf = 8
        R = Gf*self.Rzero*self.c/rho + self.Rzero
        return R

    def exactStateData(self, R):
        Gf = 8
        rho = (Gf*self.Rzero*self.c)/(R-self.Rzero)
        return rho

    # Função que cria um array de dados simulados de resistencia    
    def simulateArray(self, n):
        multiplier = []
        self.array = []
        self.arrayPerfect = []
        self.exactStateArray = []
        diferences = []
        for i in range(0, n):
            multiplier = self.simFunction(i)
            simu_sample = multiplier*self.realData(self.rho)
            perf_sample = multiplier*self.exactData(self.rho)
            self.array.append(simu_sample)
            self.arrayPerfect.append(perf_sample)
            self.exactStateArray.append(self.exactStateData(perf_sample))
            diferences.append(simu_sample - perf_sample)
        self.errorMean = np.mean(diferences)
        self.errorVar = np.var(diferences)
        pass



class ObservationError(StrainGauge):

    def __init__(self):
        pass

# Testes
"""n = 1000
R = StrainGauge()
R.simulate_array(n)
print(R.mean)

import matplotlib.pyplot as plt
plt.plot(R.arrayPerfect,'k', label = 'R - Modelo')
plt.plot(R.array,'r.', label = 'R - Observação')
plt.title('Resistência do Extensômetro')
plt.legend()
plt.show()"""
