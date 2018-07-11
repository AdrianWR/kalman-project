import numpy as np
import random

# Esta classe cria um objeto Extensometro, simulando parametros fisicos
# do instrumento. As constantes inicializaveis podem ser alteradas após
# a instanciação do objeto, sendo as principais Resistencia Inicial (Rzero),
# meia altura do extensometro (c) e raio de curvatura (rho).


#ADICIONAR erro de medida vk
class StrainGauge:

    def __init__(self, Rzero = 200, c = 0.2):
        self.Rzero  = Rzero
        self.c      = c
        #self.rho    = self.rho()
        pass

    # Gera um valor de raio de curvatura de distribuição normal
    def rho(mean = 10, std = 0.5):
        return random.gauss(mean, std)

    # Função de teste simulando variação de resistência com o tempo.
    def simFunction(t):
        return 1

    # Função que retorna o valor da Resistencia levando os erros em conta
    def realData(self, rho):
        Gf = random.gauss(8,1)
        Rzero = random.gauss(200, 10)
        #Rzero = self.Rzero
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
        #multiplier = []
        self.array = []
        self.arrayPerfect = []
        self.exactStateArray = []
        diferences = []
        for i in range(0, n):
            rho = self.rho()
            multiplier = self.simFunction(i)
            simu_sample = multiplier*self.realData(rho)
            perf_sample = multiplier*self.exactData(rho)
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

class RandomVariable(object):

    distributions = ['notRandom','gauss','uniform']

    def __init__(self, mean, std = 0, dist = 'notRandom'):
        
        if dist not in distributions:
            print('Variável ' + __str__ + ' não pode assumir distribuição do tipo ' + dist + '.')
        
        self.mean = mean
        self.std = std
        
        pass

    def __call__(self):

        if (dist == 'gaussian'):
            return random.gauss(self.mean, self.std)
        if (dist == 'uniform'):
            return random.uniform(self.mean, self.std)
        else:
            return self.mean




if __name__ == '__main__':
    R = StrainGauge()
    #random.gauss(12)
    n = RandomVariable(2,1)
    n()




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
