import json
import numpy as np
from numpy import pi, linspace, array
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


def radius_of_curvature(ellipse, points):
    a = round(ellipse.hradius,2)
    b = round(ellipse.vradius,2)
    rho_ellipse = []
    for i in range(len(points)):
        x, y = points[i]
        rho = ((a*b)**2)*(((x**2/a**4)+(y**2/b**4))**(3/2))
        rho_ellipse.append(rho.__round__(2))
    return rho_ellipse

## Distance between gauges is maintened equal
def ellipse_points(ellipse, n = 12):

    c = 0
    dtheta = 0
    C = ellipse.circumference.evalf()
    e = round(ellipse.eccentricity.evalf(),3)
    a = round(ellipse.hradius,2)
    b = round(ellipse.vradius,2)

    points = [0]*n
    points[0]        = [a, 0]
    points[n//4]     = [0, b]
    points[2*n//4]   = [-a,0]
    points[3*n//4]   = [0,-b]
    k = 1
    while (k < n//4):
        while (c < k*C//n):
            c = a*ellipeinc(dtheta, e**2)
            dtheta += 0.01
        x = round(a*np.cos(dtheta),2)
        y = round(b*np.sin(dtheta),2)
        points[k]        = [+x,+y]
        points[k+n//4]   = [-x,+y]
        points[k+2*n//4] = [-x,-y]
        points[k+3*n//4] = [+x,-y]
        k += 1
    return points

## Angle is maintened equal
def ellipse_points2(ellipse, n = 12):

    a = round(ellipse.hradius,2)
    b = round(ellipse.vradius,2)

    points = [0]*n
    #i = 0
    for i in range(0, n):
        angle = i*(2*pi/n)
        x = a*np.cos(angle)
        y = b*np.sin(angle)
        points[i] = [x.__round__(2),y.__round__(2)]
        #i += 1
    return points

# Iterations Information
n = 20
nGauges = 12
a = np.zeros(n)
b = linspace(b,1.2*b,n)


# Print ellipse objects information to file
ellipses = []
for i in range(n):
    e = Ellipse(Point(0, 0), hradius = x2.evalf(subs={y:b[i]}), vradius = b[i])
    ePoints = ellipse_points2(e, nGauges)
    data = {
        "coordinates": ePoints,
        "radius_of_curvature": radius_of_curvature(e, ePoints),
        "semiaxis": [e.hradius.__round__(2), e.vradius.__round__(2)]
    }
    ellipses.append(data)
json.dump(ellipses, open('ellipses.json', 'w'), indent=2)
print('Number of ellipses generated: '+ n.__str__() +'. Each one with '+ nGauges.__str__() +' strain gauges.')