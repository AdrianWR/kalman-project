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
# rho: radius of curvature taken from strain gauge. numpy.ndarray type
# Gf: Instrument gauge factor.                      RandomVariable type
# n: number of measurements to take                 int type   

class StrainGauge(object):

    def __init__(self, Rzero, c, rho, Gf, err = 0):

        self.Rzero      = Rzero
        self.c          = c
        self.Gf         = Gf
        self.err        = err

        for i in vars(self):
            if (type(vars(self)[i]) == int or type(vars(self)[i]) == float):
                vars(self)[i] = RandomVariable(vars(self)[i])
            else:
                pass
        
        self.rho   = rho
        pass

    # As number of samples 'n' is assigned, create samples array as object property 
    def __setattr__(self, name, value):
        self.__dict__[name] = value
        if name == 'rho':
            self.realizeArray(value)


    # Realize observable data from input parameters.
    def realizeData(self, rho):
        
        Gf    = self.Gf()
        Rzero = self.Rzero()
        c     = self.c()
        err   = self.err()

        try:            
            R = (Gf*Rzero*c/rho) + Rzero + err
            rho_measure = (Gf*Rzero*c)/(R-Rzero)
            return [rho_measure, R]
        except TypeError:
            print('Something went wrong on data realization, check variable types.')
            raise SystemExit

    # Generate n-th array with data realized from parameters
    def realizeArray(self, rho):
        n = rho.distributionArray.size
        self.realizationArray = np.zeros(n)
        self.measuredStateArray = np.zeros(n)
        for i in range(n):
            data = self.realizeData(rho.distributionArray[i])
            self.measuredStateArray[i] = data[0]
            self.realizationArray[i]   = data[1]
        pass

    # def stateFromRealization(self):
    #     n = self.realizationArray.size
    #     stateArray = np.zeros(n)
    #     for i in range(n):
    #         stateArray[i] = self.Gf()*self.Rzero()*self.c()
    #         stateArray[i] = stateArray[i]/(self.realizationArray[i] - self.Rzero())
    #     return stateArray



# This class defines a random variable data type to be used during the simulations. With its help, it's possible to
# create random controlled data based on statistic distributions and non-Random data (in the case of exact realizations).
# To generate random doubles based on the distributions, call the instance as method (__call__).

# PROPERTIES SUMMARY:
# mean: Mean value of the distribution, or exact value regarding non random data.
# std: Populational standard err of distribution. If declared to non random data, its value is ignored.
# var: Distribution variance. Ignored regarding non random data.
# dist: Distribution type to be used. May be assigned as 'gaussian', 'uniform' or 'nonRandom' (default value).

# METHODS SUMMARY:
# __call__(): Generate random data each time it's called.
# uniformLowHigh: Update 'mean' and 'std' properties to use 'low' and 'high' distribution parameters in 'uniform'.

class RandomVariable(object):

    def __init__(self, mean = 0, std = 0, dist = 'nonRandom', n = 0):
    
        _distributions = ['nonRandom','gaussian','uniform']

        if dist not in _distributions:
            print('Variável ' + __name__ + ' não pode assumir distribuição do tipo ' + dist + '.')
        
        self.mean = mean
        self.std = std
        #self.var = std**2
        self.dist = dist
        #self.distributionArray = np.array([mean])
        self.n = n
    
    def __call__(self):
        if (self.dist == 'gaussian'):
            return random.gauss(self.mean, self.std)
        elif (self.dist == 'uniform'):
            return random.uniform(self.mean-self.std*np.sqrt(3),self.mean+self.std*np.sqrt(3))
        else:
            return self.mean
        pass
    
    def __setattr__(self, name, value):
        self.__dict__[name] = value
        if name == 'n' and value > 0:
            self.distributionArray = np.zeros(value)
            for i in range(value):
                self.distributionArray[i] = self()
        elif name == 'std':
            self.var = value**2

    def uniformLowHigh(self, low, high):
        
        if (self.dist == 'uniform'):
            self.mean = (low + high)/2
            self.std = (high-low)/(np.sqrt(12))
            self.var = (self.std)**2
        else:
            raise TypeError("Distribution must be 'uniform' to call this method.")

    # def realizeData(self):

    #     if (self.dist == 'gaussian'):
    #         return random.gauss(self.mean, self.std)
    #     elif (self.dist == 'uniform'):
    #         return random.uniform(self.mean-self.std*np.sqrt(3),self.mean+self.std*np.sqrt(3))
    #     else:
    #         return self.mean
    #     pass
 

# Inherited class from RandomVariable.
# Calculate the Aproximation Error based on two instances of StrainGauge class.
# As a result, can be called as RandomVariable, with its own 'mean' and 'std' (gaussian distribution).

class ApproximationError(RandomVariable):
    
    def __init__(self, strainGauge1 = 0, strainGauge2 = 0):
        
        if (strainGauge1 == 0 and strainGauge2 == 0):
            RandomVariable.__init__(self)
        elif (strainGauge2.rho != strainGauge1.rho):
            print('Random state variable is unequal on models. Assuming distribution of first argument.')
            strainGauge2.rho = strainGauge1.rho
        else:
            realizationArray = strainGauge1.realizationArray - strainGauge2.realizationArray
            RandomVariable.__init__(self, realizationArray.mean(), realizationArray.std(),'uniform')
            pass
        
        pass

# ##################
# Class Testing Area
# ##################
if __name__ == '__main__':
    
    rho   = RandomVariable(10, 0.5, 'gaussian', 5)
    rho2  = RandomVariable(10, 0.5, 'gaussian', 5)
    Rzero = RandomVariable(2,1,'gaussian')
    c     = RandomVariable(2)
    Gf    = RandomVariable(8,1,'gaussian')
    dev   = RandomVariable(dist = 'uniform')
    dev.uniformLowHigh(-0.05,0.05)


    R1 = StrainGauge(Rzero, c, rho, Gf, err = 0)
    R2 = StrainGauge(3,2,rho2,8, err = dev)
    eps = ApproximationError(R1, R2)

    import json


    print('ok')
