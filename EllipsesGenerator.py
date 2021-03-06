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
    a = ellipse.hradius
    b = ellipse.vradius
    rho_ellipse = []
    for i in range(len(points)):
        x, y = points[i]
        rho = ((a*b)**2)*(((x**2/a**4)+(y**2/b**4))**(3/2))
        rho = round(rho, 6)
        rho_ellipse.append(rho)
    return rho_ellipse

## Distance between gauges is maintened equal
def ellipse_points(ellipse, n = 12):

    c = 0
    dtheta = 0
    C = ellipse.circumference.evalf()
    e = ellipse.eccentricity.evalf()

    a = ellipse.hradius
    b = ellipse.vradius

    points = [0]*n
    points[0]        = [a.__round__(6), 0]
    points[n//4]     = [0, b.__round__(6)]
    points[2*n//4]   = [-a.__round__(6),0]
    points[3*n//4]   = [0,-b.__round__(6)]
    k = 1
    while (k < n//4):
        while (c < k*C//n):
            c = a*ellipeinc(dtheta, round(e**2, 16))    # Not rounding causes TypeError at function
            dtheta += 0.01
        x = a*np.cos(dtheta)
        y = b*np.sin(dtheta)
        x = x.__round__(6)
        y = y.__round__(6)
        points[k]        = [+x,+y]
        points[k+n//4]   = [-x,+y]
        points[k+2*n//4] = [-x,-y]
        points[k+3*n//4] = [+x,-y]
        k += 1
    return points

## Angle is maintened equal
def ellipse_points2(ellipse, n = 12):

    a = ellipse.hradius
    b = ellipse.vradius

    points = [0]*n
    for i in range(0, n):
        angle = i*(2*pi/n)
        x = a*np.cos(angle)
        y = b*np.sin(angle)
        x = x.__round__(6)
        y = y.__round__(6)
        points[i] = [x, y]
    return points

# Iterations Information
n = 250
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
        "semiaxis": [e.hradius.__round__(6), e.vradius.__round__(6)]
    }
    ellipses.append(data)
    print('Ellipse number ' + str(i+1) + ' generated.')
json.dump(ellipses, open('ellipses.json', 'w'), indent=2)
print('Number of ellipses generated: '+ n.__str__() +'. Each one with '+ nGauges.__str__() +' strain gauges.')