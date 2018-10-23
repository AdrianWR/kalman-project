import matplotlib.pyplot as plt
import numpy as np
import json

def draw_ellipse(ellipse):
    
    a, b = ellipse['semiaxis']
    x,y = np.transpose(np.array(ellipse['coordinates']))
    theta = np.arctan2(y,x)
    rho = ellipse['radius_of_curvature']
    u,v = [,]
    # u,v = [rho*sin(theta), rho*cos(theta)]

    t = np.linspace(0, 2*np.pi, 100)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x,y)
    ax.plot(a*np.cos(t), b*np.sin(t), 'g--')
    ax.set_xlim(-160, 160)
    ax.set_ylim(-160, 160)
    
    

    plt.quiver()

    plt.show()

data = json.load(open('ellipses.json','r'))
rc = np.array(data[0]['radius_of_curvature'])
angle = np.linspace(0, 2*np.pi, 12)

n = len(rc)
rho_a = np.mean([rc[0], rc[n//2]])
rho_b = np.mean([rc[n//4], rc[3*n//4]])

a = np.cbrt(rho_a*rho_b**2)
b = np.cbrt(rho_b*rho_a**2)

coordinates = [[a*np.cos(t), b*np.sin(t)] for t in angle]
draw_ellipse(data[0])

print('Done.')