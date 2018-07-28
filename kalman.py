# Neste módulo encontram-se funções básicas que auxiliam
# o desenvolvimento de filtros de Kalman

import numpy as np

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

                self.K = np.array([0])

        # As funções h e HK podem ser alteradas no programa de filtragem.
        # Por default, observará o estado da variável x e sua derivada.

        def f(self, x):
                return x

        def h(self, x):
               return x

        def HK(self, x):
               return x

        def propagate(self):

                x = self.x
                P = self.P
                Fk = self.Fk
                Q = self.Q

                self.x = self.f(x)
                self.P = Fk*P*Fk.T + Q
                #self.K = P*HK(x).T
                #self.K = self.K/(HK(x)*P*HK(x).T+R)


        def update(self, ym):

                x = self.x
                P = self.P
                h = self.h
                HK = self.HK
                R = self.R

                I = np.array([1.0])
                self.K = (P*HK(x).T)/((HK(x)*P*HK(x).T+R))
                self.x = self.x + self.K*(ym-h(self.x))
                self.P = (I-self.K*HK(x))*self.P
                #self.P = (self.P+self.P.T)/2
                #print(P)

        def filter(self, observer):

                n = len(observer)
                self.signal = []
                for i in range(0,n):
                        self.propagate()
                        self.update(observer[i])
                        self.signal.append(self.x[0])
                return self.signal