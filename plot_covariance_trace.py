import matplotlib.pyplot as plt
import numpy as np
import json

data = json.load(open("covariance_trace.json","r"))

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=12)

# Covariance Trace
plt.figure(0, figsize = (10,6))
plt.title('Traco da Matriz de Covariancias de Estado')
ax0 = plt.subplot(1,1,1)
ax0.plot(data[0]['trace'], label = data[0]['legend'], c = 'r', linewidth=2)
ax0.plot(data[1]['trace'], label = data[1]['legend'], c = 'm', linewidth=2)
ax0.set_xlabel('Amostras')
ax0.set_ylabel('Traco da Covariancia ' + r'$(\Omega^2)$')
#ax0.set_yscale('log')
#ax0.set_yticks([10,100,1000])
plt.legend()
#plt.show()
plt.savefig("./media/covariance_trace.png", dpi=96)
print("Figure saved.")
