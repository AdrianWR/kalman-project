import matplotlib.pyplot as plt
import numpy as np
import json

trueData = json.load(open("ellipses.json","r"))
nSamples = trueData.__len__()
nGauges = trueData[0]['radius_of_curvature'].__len__()
data5 = json.load(open("odr_reconstruction_5.json","r"))
data6 = json.load(open("odr_reconstruction_6.json","r"))

for i in [0, 50, 150, 249]:

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set(xlim=(-160, 160), ylim=(-160, 160))

    t = np.linspace(0, 2*np.pi, 100)
    a, b = trueData[i]['semiaxis']
    x, y = [a*np.cos(t), b*np.sin(t)]
    line = ax.plot(x, y, lw=2)[0]
    line.set_color('blue')
    line.set_linestyle('--')
    line.set_label('Geometria Verdadeira')

    a, b = data5['odr_semiaxis'][i]
    x, y = [a*np.cos(t), b*np.sin(t)]
    line2 = ax.plot(x, y, lw=2)[0]
    line2.set_color('red')
    line2.set_linestyle('-')
    line2.set_label('Reconstrução - Extensometros Isolados')

    a, b = data6['odr_semiaxis'][i]
    x, y = [a*np.cos(t), b*np.sin(t)]
    line2 = ax.plot(x, y, lw=2)[0]
    line2.set_color('orange')
    line2.set_linestyle('-')
    line2.set_label('Reconstrução - Compensação por Vizinhos')

    plt.title('Reconstrucao Geometrica Toracica')
    ax.legend(loc = 1)
    plt.savefig("./media/reconstruction_frame_" + str(i), dpi=96)
    
    #plt.draw()
    #plt.show()

#data = json.load(open("covariance_trace.json","r"))

# Reconstructions
#for i in [0, 50, 150, 249]:
        #path = imgDirectory + 'reconstruction_frame_' + str(i) + '.png'
        #draw_ellipse3(trueData[i], data5[i])