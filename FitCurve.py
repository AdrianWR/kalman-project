import numpy as np
import json

data = json.load(open('ellipses.json','r'))
rc = np.array(data[0]['radius_of_curvature'])
angle = np.linspace(0, 2*np.pi, 12)

n = len(rc)
rho_a = np.mean([rc[0], rc[n//2]])
rho_b = np.mean([rc[n//4], rc[3*n//4]])

a = np.cbrt(rho_a*rho_b**2)
b = np.cbrt(rho_b*rho_a**2)

coordinates = [[a*np.cos(t), b*np.sin(t)] for t in angle]

print('Done.')