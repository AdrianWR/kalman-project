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

    # Função de teste simulando variação de resistência com o tempo.
    def simFunction(t):
        return t
        
    # Função que retorna o valor da Resistencia levando os erros em conta
    def simulate(self):
        Gf = random.gauss(8,1)  # O coeficiente de extensometria é Gf, com mu = 8 e sigma = 1
        R = Gf*self.Rzero*self.c/self.rho + self.Rzero
        return R

    # Função que retorna o valor da Resistencia sem erros associados
    def perfect(self):
        Gf = 8
        R = Gf*self.Rzero*self.c/self.rho + self.Rzero
        return R

    # Função que cria um array de dados simulados de resistencia    
    def simulate_array(self, n):
        self.array = []
        self.arrayPerfect = []
        for i in range(0, n):
            self.array.append(self.simFunction(self.simulate()))
            #self.array.append(self.simulate())
            self.arrayPerfect.append(self.simFunction(self.perfect()))
        self.mean = np.mean(self.array)
        self.var = np.var(self.array)

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
