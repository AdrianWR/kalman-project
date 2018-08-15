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
                self.P = Fk*P*Fk.T + Q


        def update(self, z):

                x = self.x
                P = self.P
                R = self.R
                H = self.H(x)
                y = z - self.h(x)

                I = np.array([1.0])
                K = P*H.T
                K = K/(H*P*H.T+R)
                self.x = x + K*y
                self.P = (I-K*H)*P
                # Joseph Form
                #self.P = (I-self.K*HK(x))*P*(I-self.K*HK(x)).T + P*self.Q*P.T
                #self.P = (self.P+self.P.T)/2
                print(self.P)

        def filterSample(self, z):

                self.propagate()
                self.update(z)
                return self.x

        def filterSampleArray(self, observer):

                n = len(observer)
                signal = []
                for i in range(0,n):
                        x = self.filterSample(observer[i])
                        signal.append(x[0])
                return np.array(signal)