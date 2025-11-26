import numpy as np


epsilon = 1e-8  

def weight(client_weight, u_gepi, u_gale, u_lepi, u_lale, epoch):
    client_weight += u_gepi[:, epoch] / ((u_lepi[:, epoch]+ epsilon) * (u_lale[:, epoch] + u_gale[:, epoch] + epsilon))
    client_weight /= client_weight.sum()
    return np.clip(client_weight, a_min=1e-3, a_max=None)