import numpy as np
import matplotlib.pyplot as plt

def orr_poisson(k,lam):
    k_fact = np.math.factorial(k)
    return ( (lam**k) * np.exp(-lam) ) / k_fact

catch = np.array([])
lam = 5
for i in range(0,100):
    out = orr_poisson(i,lam)
    catch = np.append(catch,out)
    
plt.plot(catch)
