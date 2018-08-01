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

    def __init__(self, Rzero, c, rho, Gf, err = 0, n = 0):

        self.Rzero      = Rzero
        self.c          = c
        self.rho        = rho
        self.Gf         = Gf
        self.err  = err

        for i in vars(self):
            if (type(vars(self)[i]) == int or type(vars(self)[i]) == float):
                vars(self)[i] = RandomVariable(vars(self)[i])
        #   elif (type(vars(self)[i]) != RandomVariable or type(vars(self)[i]) != ApproximationError):
        #       print('Input ' + vars(self)[i] + " must be either of 'int' or 'RandomVariable' type.")
        #   else:
        #       raise SystemExit
        
        self.n     = n

    # As number of samples 'n' is assigned, create samples array as object property 
    def __setattr__(self, name, value):
        self.__dict__[name] = value
        if name == 'n':
            try:
                if type(value) == int:
                    if value <= 0:
                        raise ValueError('The number of samples must be a positive integer.')
                    else:
                        self.realizeArray(value)
                else:
                    raise TypeError('The number of samples must be an integer value.')
            except Exception as error:
                print(repr(error))
                raise SystemExit


    # Realize observable data from input parameters.
    def realizeData(self, rho):
        try:
            R = (self.Gf()*self.Rzero()*self.c()/rho) + self.Rzero()
            R *= 1 + self.err()
            return np.array([rho, R])
        except TypeError:
            print('Something went wrong, check variable types.')

    # Simulation of the multiplier function. May be changed from outside.
    def simFunction(self, t):
        return 1

    # Generate n-th array with data realized from parameters
    def realizeArray(self, n):
        
        # if type(rho) == numpy.ndarray:
        #     if rho.size != n:
        #         raise ValueError('Size of rho array must be equal to samples number.')
        #         break
        #     else:
        #         pass

        self.realizationArray = np.zeros(n)
        self.stateArray = np.zeros(n)
        for i in range(0, n):
            if (type(rho) == RandomVariable):
                data = self.realizeData(rho())
            else:
                data = self.realizeData(rho[i])
            self.stateArray[i]       = data[0]
            self.realizationArray[i] = data[1]
            #self.array[i] = self.simFunction(i)*self.realizeData()[1]


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

    def __init__(self, mean = 0, std = 0, dist = 'nonRandom'):
    
        _distributions = ['nonRandom','gaussian','uniform']

        if dist not in _distributions:
            print('Variável ' + __name__ + ' não pode assumir distribuição do tipo ' + dist + '.')
        
        self.mean = mean
        self.std = std
        self.var = std**2
        self.dist = dist

    def __call__(self):

        if (self.dist == 'gaussian'):
            return random.gauss(self.mean, self.std)
        elif (self.dist == 'uniform'):
            return random.uniform(self.mean-self.std*np.sqrt(3),self.mean+self.std*np.sqrt(3))
        else:
            return self.mean
        pass

    def uniformLowHigh(self, low, high):
        
        if (self.dist == 'uniform'):
            self.mean = (low + high)/2
            self.std = (high-low)/(np.sqrt(12))
            self.var = (self.std)**2
        else:
            raise TypeError("Distribution must be 'uniform' to call this method.")

# Inherited class from RandomVariable.
# Calculate the Aproximation Error based on two instances of StrainGauge class.
# As a result, can be called as RandomVariable, with its own 'mean' and 'std' (gaussian distribution).

class ApproximationError(RandomVariable):
    
    def __init__(self, strainGauge1, strainGauge2):
        
        if (strainGauge2.rho != strainGauge1.rho):
            print('Random state variable is unequal on models. Assuming distribution of first argument.')
            strainGauge2.rho = strainGauge1.rho
        elif strainGauge1.n != strainGauge2.n:
            print('Number of samples is unequal on models. Assuming samples number of first argument.')
            strainGauge2.n = strainGauge1.n
        else:
            pass
        
        n = strainGauge1.n
        rhoSamples = np.zeros(n)
        for i in range(n):
            rhoSamples[i] = strainGauge1.rho()
        
        strainGauge1.rho = rhoSamples
        strainGauge2.rho = rhoSamples

        self.realizationArray = strainGauge1.realizationArray - strainGauge2.realizationArray
        RandomVariable.__init__(self, self.realizationArray.mean(), self.realizationArray.std(),'gaussian')
        pass

# ##################
# Class Testing Area
# ##################
if __name__ == '__main__':
    
    Rzero = RandomVariable(2,1,'gaussian')
    c     = RandomVariable(2)
    rho   = RandomVariable(10, 0.5, 'gaussian')
    Gf    = RandomVariable(8,1,'gaussian')
    dev   = RandomVariable(dist = 'uniform')
    dev.uniformLowHigh(-5,5)

    n = 50
    #c = 'paçoca'
    R1 = StrainGauge(Rzero, c, rho, Gf, n = 50)
    R2 = StrainGauge(3,2,10,8, dev, n = 50)
    eps = ApproximationError(R1, R2)

    print('ok')