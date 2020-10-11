import numpy as np

def phi(x, Pe) :

    return (np.exp(Pe*x) - 1) / (np.exp(Pe) - 1)