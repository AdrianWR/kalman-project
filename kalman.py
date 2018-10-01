# Neste módulo encontram-se funções básicas que auxiliam
# o desenvolvimento de filtros de Kalman

import numpy as np
from numpy import dot, eye, array

### DEFINIÇÕES RELEVANTES
# As variáveis inseridas na inicialização da classe
# devem estar em formato Numpy Array. Métodos de validação
# serão inseridos posteriormente.

class ExtendedKalmanFilter(object):

        def __init__(self, x0, P0, Fk, R, Q):

                self.x = x0
                self.P = P0
                self.Fk = Fk
                self.Q = Q
                self.R = R

        # As funções h e H podem ser alteradas no programa de filtragem.
        # Por default, observará o estado da variável x e sua derivada.

        def f(self, x):
                return x

        def h(self, x):
               return x

        def H(self, x):
               return x

        def propagate(self):

                x = self.x
                P = self.P
                Fk = self.Fk
                Q = self.Q

                self.x = self.f(x)
                self.P = dot(dot(Fk,P),Fk.T) + Q


        def update(self, z):
                
                I = eye(len(self.x))

                x = self.x
                P = self.P
                R = self.R
                H = self.H(x)*I
                y = z - self.h(x)

                K = dot(P, H.T)
                K = K/(dot(dot(H,P),H.T)+R)
                self.x = x + dot(K,y)
                self.P = dot(I-dot(K,H),P)
                # Joseph Form
                #self.P = (I-self.K*HK(x))*P*(I-self.K*HK(x)).T + P*self.Q*P.T
                #self.P = (self.P+self.P.T)/2
                #print(self.x)

        def filterSample(self, z):

                self.propagate()
                self.update(z)
                return [self.x, self.P]

        def filterSampleArray(self, observer):

                n = len(observer)
                signal = []
                for i in range(0,n):
                        x = self.filterSample(observer[i])
                        signal.append(x)
                return np.array(signal)
                #[number of sample][x or P][sensor row]

class Result(ExtendedKalmanFilter):

        def __init__(self, kalmanFilter, observer):
                result = kalmanFilter.filterSampleArray(observer)
                self.x = result[:,0]
                self.P = result[:,1]
                pass

        def __call__(self):
                return self.x
