
import numpy as np
from numpy.random import default_rng

def create_grid_coords(grid_size=(160,160,160),return_cartesian_coords=False,a_vectors=np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])):
    
    ngxf, ngyf, ngzf = grid_size

    xx,yy,zz = np.meshgrid(np.arange(0,1,1/ngxf),np.arange(0,1,1/ngyf),np.arange(0,1,1/ngzf),indexing='ij')

    fcoords = np.vstack([xx.ravel(),yy.ravel(),zz.ravel()]).T
    
    if return_cartesian_coords:
        return np.dot(fcoords, a_vectors)
    else: return fcoords

def sample_charge(chg,sigma,n_samples,uniform_ratio,seed=42):

    rng = default_rng(42)

    chg = chg.ravel()

    n_prob = int(np.ceil(n_samples * (1 - uniform_ratio)))
    n_uniform = n_samples - n_prob

    prob_chg = np.exp(-((1 / chg) ** 2) / (2 * sigma ** 2)) / ((2 * np.pi * (sigma ** 2)) ** 0.5)
    prob_chg /= sum(prob_chg)

    selected_index = np.array([])
    m = 1.0
    while len(selected_index) < n_samples:
        selected_chg = rng.choice(np.arange(len(chg)),size=int(np.ceil(n_prob * m)),p=prob_chg,replace=False)
        selected_uniform = rng.choice(np.arange(len(chg)), size=n_uniform, replace=False)
        selected_index = np.unique(np.append(selected_chg, selected_uniform))      
        m += 0.1

    rng.shuffle(selected_index)

    return selected_index[:n_samples]