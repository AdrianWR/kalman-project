import numpy as np
import json
from numpy import pi, linspace
#from sympy import *
from sympy.geometry import Point, Ellipse
from sympy.abc import t
from subprocess import check_output


chestData = check_output(["Rscript","bodydata.r"], shell = True)
chestData = chestData.decode("UTF-8")
chestData = json.loads(chestData.splitlines()[0])
rc_min = chestData[0]
rc_max = chestData[1]
a = int(chestData[2]['mean'])
b = int(chestData[3]['mean'])

n = 12

# MOMENTO 1
e1 = Ellipse(Point(-a, 0), hradius = a, vradius = b)
parametric = e1.arbitrary_point(t)

samples = linspace(0, 2*pi, n)
e1_points = []
for i in samples:
    e1_points.append(parametric.evalf(subs={t:i}))

