import numpy as np
import random

# Esta classe cria um objeto Extensometro, simulando parametros fisicos
# do instrumento. As constantes inicializaveis podem ser alteradas após
# a instanciação do objeto, sendo as principais Resistencia Inicial (Rzero),
# meia altura do extensometro (c) e raio de curvatura (rho).


#ADICIONAR erro de medida vk
# StrainGauge class will be capable to receive random variables in __init__
class StrainGauge:

    def __init__(self, Rzero, c, rho, Gf):

        self.Rzero  = Rzero
        self.c      = c
        self.rho    = rho
        self.Gf     = Gf
        pass

    # Gera um valor de raio de curvatura de distribuição normal
    def rho(self, mean = 10, std = 0.5):
        return random.gauss(mean, std)

    # Função de teste simulando variação de resistência com o tempo.
    def simFunction(self, t):
        return 1

    # Função que retorna o valor da Resistencia levando os erros em conta
    def realizeData(self):
        try:
            return self.Gf()*self.Rzero()*self.c()/self.rho() + self.Rzero()
        except TypeError:
            for i in vars(self):
                if type(vars(self)[i]) == int:
                    RandomVariable(vars(self)[i])
            
            print('No!')


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

class RandomVariable():

    def __init__(self, mean, std = 0, dist = 'notRandom'):
        
        distributions = ['notRandom','gaussian','uniform']

        if dist not in distributions:
            print('Variável ' + __name__ + ' não pode assumir distribuição do tipo ' + dist + '.')
        
        self.mean = mean
        self.std = std
        self.dist = dist

        pass

    def __call__(self):

        if (self.dist == 'gaussian'):
            return random.gauss(self.mean, self.std)
        elif (self.dist == 'uniform'):
            return random.uniform(self.mean, self.std)
        else:
            return self.mean



if __name__ == '__main__':
    
    #random.gauss(12)
    Rzero = RandomVariable(2,1,'gaussian')
    rho   = RandomVariable(10, 0.5, 'gaussian')
    Gf    = RandomVariable(8,1,'gaussian')
    #c     = RandomVariable(2)
    c = 3
    R = StrainGauge(Rzero, c, rho, Gf)

