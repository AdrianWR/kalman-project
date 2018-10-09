#import numpy as np
#import sympy
import json
from numpy import pi, linspace
from scipy.special import ellipeinc
from sympy import Symbol, solve, sqrt
from sympy.geometry import Point, Ellipse
from subprocess import check_output


chestData = check_output(["Rscript","bodydata.r"], shell = True)
chestData = chestData.decode("UTF-8")
chestData = json.loads(chestData.splitlines()[0])
rc_min = chestData[0]
rc_max = chestData[1]
a = int(chestData[2]['mean'])
b = int(chestData[3]['mean'])


# MOMENT 0
e0 = Ellipse(Point(0, 0), hradius = a, vradius = b)
P = e0.circumference
E = e0.eccentricity
theta = Symbol('theta')

x = Symbol('x')
y = Symbol('y')
P = pi*(3*(x+y)-sqrt((3*x + y)*(x + 3*y))) - P
x1, x2 = solve(P,x)

def theta_K(ellipse):

    c = 0
    dtheta = 0
    C = ellipse.circumference.evalf()
    e = round(ellipse.eccentricity.evalf(),3)

    radlist = []
    k = 0
    while (k < 4):
        while (c < k*C/12):
            c = 4*a*ellipeinc(dtheta, e**2)
            dtheta += 0.01
        radlist.append(round(dtheta,2))
        k += 1
    return radlist

theta_K(e0)
# Iterations
n = 20
nGauges = 12
a = np.zeros(n)
b = linspace(b,1.2*b,n)

e_points = []
for i in range(n):

    #x = x2.evalf(subs={y:b[i]})
    #y = b[i]
    e = Ellipse(Point(0, 0), hradius = x2.evalf(subs={y:b[i]}), vradius = b[i])
    
    parametric = e.arbitrary_point(theta)
    samples = linspace(0, 2*pi, nGauges)
    e_points.append([])
    for j in samples:
        e_points[i].append(parametric.evalf(subs={theta:j}))

