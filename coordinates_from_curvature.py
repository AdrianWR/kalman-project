import sympy
from sympy import symbols, solve, sin, cos

K, a, b, t = symbols('K 144 126 t')
# K = a*b/(a**2*(sin(t))**2+b**2*(cos(t))**2)
# K = a*b/(a**2*sin(t)**2 + b**2*cos(t)**2)**(3/2)
K = a*b*(a**2*sin(t)**2 + b**2*cos(t)**2)**(-1.5)
K = 144*126*(144**2*sin(t)**2 + 126**2*cos(t)**2)**(-1.5)
print('Done.')