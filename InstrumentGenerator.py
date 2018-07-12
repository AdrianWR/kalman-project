import numpy as np
import random

# Esta classe cria um objeto Extensometro, simulando parametros fisicos
# do instrumento. As constantes inicializaveis podem ser alteradas após
# a instanciação do objeto, sendo as principais Resistencia Inicial (Rzero),
# meia altura do extensometro (c) e raio de curvatura (rho).


#ADICIONAR erro de medida vk

# StrainGauge class will be capable to receive random variables in __init__
# If a int is inputed, it'll be transformed into a random variable with std = 0
class StrainGauge:

    def __init__(self, Rzero, c, rho, Gf, n = 0):

        self.Rzero  = Rzero
        self.c      = c
        self.rho    = rho
        self.Gf     = Gf
        self.n      = n

        for i in vars(self):
                if type(vars(self)[i]) == int:
                    vars(self)[i] = RandomVariable(vars(self)[i])
                elif type(vars(self)[i]) != RandomVariable:
                    print('Input ' + vars(self)[i] + " must be either of 'int' or 'RandomVariable' type.")
                    raise SystemExit
                else:
                    pass
        
        self.array = []
        pass

    

    #def __setattr__(self, n, x):
    #    print('n changes')
    #    if (self.n.mean != 0):
    #        self.realizeArray(self.n.mean)
    #    pass

    # Realize observable data from input parameters.
    def realizeData(self):
        try:
            return self.Gf()*self.Rzero()*self.c()/self.rho() + self.Rzero()
        except TypeError:
            print('Something went wrong, check variable types.')

    # Return state variable rho from observable data R and input parameters
    #def realizeStateData(self, R):
    #    return (self.Gf*self.Rzero*self.c)/(R-self.Rzero)

    # Simulation of the multiplier function. Could be change from outside.
    def simFunction(self, t):
        return 1

    # Generate n-th array with data realized from parameters
    def realizeArray(self, n):        
        for i in range(0, n):
            multiplier = self.simFunction(i)
            self.array.append(multiplier*self.realizeData())
            #self.stateArray.append(multiplier*self.realizeStateData())
        pass

class ApproximationError(StrainGauge):
    
    def __init__():
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
        pass
        



if __name__ == '__main__':
    
    #random.gauss(12)
    Rzero = RandomVariable(2,1,'gaussian')
    rho   = RandomVariable(10, 0.5, 'gaussian')
    Gf    = RandomVariable(8,1,'gaussian')
    c     = RandomVariable(2)
    n = 50
    #c = 'paçoca'
    R = StrainGauge(Rzero, c, rho, Gf, n)
    print('ok')