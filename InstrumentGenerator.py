import numpy as np
import random

# This class creates a strain gauge instrument model, simulating its physical properties.
# Properties can be given as parameters after instantiation, but is heavily advised
# to initialize them at object's creation. If Rzero, c, rho and Gf are assigned as 'int'
# type variables, __setattr__ special method will transform them to RandomVariable type
# with mean = value and std = 0.

# PROPERTIES SUMMARY
# Rzero: Initial value of strain gauge resistance.  RandomVariable type
# c: Half height of strain gauge.                   RandomVariable type
# rho: radius of curvature taken from strain gauge. RandomVariable type
# Gf: Instrument gauge factor.                      RandomVariable type
# n: number of measurements to take                 int type   

class StrainGauge(object):

    def __init__(self, Rzero, c, rho, Gf, n = None):

        self.Rzero  = Rzero
        self.c      = c
        self.rho    = rho
        self.Gf     = Gf
        for i in vars(self):
                if type(vars(self)[i]) == int:
                    vars(self)[i] = RandomVariable(vars(self)[i])
                elif type(vars(self)[i]) != RandomVariable:
                    print('Input ' + vars(self)[i] + " must be either of 'int' or 'RandomVariable' type.")
                    raise SystemExit
                else:
                    pass
        
        self.n     = n
        self.array = []
        pass    

    # As number of samples 'n' is assigned, create samples array as object property 
    def __setattr__(self, name, value):
        self.__dict__[name] = value
        if name == 'n':
            try:
                if value == None:
                    pass
                elif type(value) == int:
                    print('n changes')
                    self.realizeArray(value)
                else:
                    print('Number of samples must be an integer value.')
                    raise SystemExit
            except:
                print("Something went wrong. Try 'n' as 'int' and 'n' > 0")
                raise SystemExit 
        pass

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
    
    def __init__(self):
        pass

class RandomVariable():

    def __init__(self, mean, std = 0, dist = 'notRandom'):
        
        _distributions = ['notRandom','gaussian','uniform']

        if dist not in _distributions:
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
        

# Class Testing Area
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