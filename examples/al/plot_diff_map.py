import numpy as np
import matplotlib.pyplot as plt

from pymatgen.io.vasp.outputs import Chgcar
from ase.io import read

import sys
sys.path.append('../../')
sys.path.append('../../jlgridfingerprints')
from jlgridfingerprints.fingerprints import JLGridFingerprints
from jlgridfingerprints.tools import create_grid_coords,sample_charge

from sklearn import metrics

import time
import os
from tqdm import tqdm
import pickle

def plot_corr_scatter(label,y_true,y_pred,savefig=False,model_name='',show=False):
    
    r2_score = metrics.r2_score(y_true,y_pred)
    mae_score = metrics.mean_absolute_error(y_true,y_pred)
    mse_score = metrics.mean_squared_error(y_true,y_pred)
    rmse_score = metrics.mean_squared_error(y_true,y_pred,squared=False)
    maxae_score = metrics.max_error(y_true,y_pred)
    
    min_ax = min(y_true.min(),y_pred.min())*0.9
    max_ax = max(y_true.max(),y_pred.max())*1.1 
    
    fig, ax = plt.subplots(figsize=(6,6))
        
    x0, x1 = min_ax,max_ax
    lims = [max(x0, x0), min(x1, x1)]
    ax.plot(lims, lims, c='tab:blue',ls='--',zorder=1,lw=1)
    
    ax.scatter(y_true,y_pred,marker='x',color='tab:blue',linewidth=1.5,edgecolors=None,alpha=1.0)

    ax.set_xlabel('DFT '+label,fontsize=16)
    ax.set_ylabel('ML '+label,fontsize=16)
    
    ax.text(s=r'R$^2$ ='+f' {r2_score: .6f}',x=0.1*(max_ax-min_ax)+min_ax,y=0.9*(max_ax-min_ax)+min_ax,fontsize=16,ha='left',va='top')
    ax.text(s=r'RMSE ='+f' {rmse_score: .6f}',x=0.5*(max_ax-min_ax)+min_ax,y=0.1*(max_ax-min_ax)+min_ax,fontsize=16,ha='left')
    ax.text(s=r'MAE  ='+f' {mae_score: .6f}',x=0.5*(max_ax-min_ax)+min_ax,y=0.175*(max_ax-min_ax)+min_ax,fontsize=16,ha='left')
    ax.text(s=r'MaxAE ='+f' {maxae_score:.6f}',x=0.5*(max_ax-min_ax)+min_ax,y=0.25*(max_ax-min_ax)+min_ax,fontsize=16,ha='left')
    #ax.text(s=r'MSE  ='+f' {mse_score: .8f}',x=0.5*(max_ax-min_ax)+min_ax,y=0.05*(max_ax-min_ax)+min_ax,fontsize=16,ha='left')

    ax.tick_params(labelsize=14,direction='in', top=True,right=True)

    ax.set_xlim((min_ax,max_ax))
    ax.set_ylim((min_ax,max_ax))

    ax.set_title(model_name,fontsize=14,ha='center')

    fig.tight_layout()
    
    if savefig:
        fig.savefig('figures/'+model_name+'_scatter.png',dpi=300,bbox_inches='tight')

    if show:
        plt.show()
    plt.close()

if not os.path.exists("figures"):
    os.makedirs("figures")

settings = {'rcut': 4.08,
            'nmax': [15,6],
            'lmax': 6,
            'alpha': [7.875386069413652,5.875090883472657],
            'beta': [3.6238075908648106,1.7505953204305842],
            'rmin': -0.74,
            'species': ['Al'],
            'body': '1+2',
            'periodic': True,
            'double_shifted': True,
            }

jl = JLGridFingerprints(**settings)

scf_path = "scf_data"
model_name = 'linear_model_chg'

frame = 0

atoms = read(scf_path + f"/test_data/{frame}/POSCAR")
vol = atoms.get_volume()
chgcar = Chgcar.from_file(scf_path + f"/test_data/{frame}/CHGCAR")

ngxf,ngyf,ngzf = chgcar.data['total'].shape
zplane = 0.5

frac_points = np.zeros((ngxf*ngyf,3))
chg_dft_points = np.zeros((ngxf*ngyf))

with tqdm(total=ngxf*ngyf) as pbar:
    io = 0
    for nx in range(ngxf):
        for ny in range(ngyf):

            frac_points[io] = (nx/ngxf,ny/ngyf,zplane)
            chg_dft_points[io] = chgcar.data['total'][nx,ny,int(zplane*ngzf)] / vol

            io += 1
            pbar.update(1)

cart_positions = np.dot(frac_points,atoms.get_cell().array)

t_init = time.time()
descriptors = jl.create(atoms, cart_positions)
print(f'Time for {cart_positions.shape[0]} descriptors: {(time.time()-t_init):>7.3f} sec')

model = pickle.load(open(f'scikit_{model_name}.p','rb'))
chg_pred_points = model.predict(descriptors)
chg_diff_points = chg_pred_points - chg_dft_points

diff_min = chg_diff_points.min()
diff_max = chg_diff_points.max()

vmin = -max(abs(diff_min),abs(diff_max))
vmax = -vmin

fig, ax = plt.subplots(figsize=(6,6),dpi=100)
imshow_ax = ax.imshow(chg_dft_points.reshape((ngxf,ngyf),order='C'),cmap='cividis',origin='lower')
ax.set_xlim((0,ngxf-1))
ax.set_ylim((0,ngyf-1))
ax.set_xticks([tick for tick in range(0,ngxf,20)])
ax.set_yticks([tick for tick in range(0,ngyf,20)])
cbar = fig.colorbar(imshow_ax, ax=ax, extend='both',shrink=0.8)
cbar.minorticks_on()
ax.grid(color='silver', linestyle='--', linewidth=0.5,alpha=0.3)
ax.tick_params(labelsize=10,direction='in',top=True,left=True,bottom=True,right=True)
cbar.ax.tick_params(labelsize=10,which='major',direction='in')
cbar.ax.tick_params(which='minor',right=False)
ax.set_title(r'DFT charge density (e/$\rm \AA^3$)',fontsize=12,pad=10)
fig.tight_layout()
fig.savefig('figures/'+f'dft_chg_map_test_{frame}.png',dpi=300,bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(6,6),dpi=100)
imshow_ax = ax.imshow(chg_pred_points.reshape((ngxf,ngyf),order='C'),cmap='cividis',origin='lower')
ax.set_xlim((0,ngxf-1))
ax.set_ylim((0,ngyf-1))
ax.set_xticks([tick for tick in range(0,ngxf,20)])
ax.set_yticks([tick for tick in range(0,ngyf,20)])
cbar = fig.colorbar(imshow_ax, ax=ax, extend='both',shrink=0.8)
cbar.minorticks_on()
ax.grid(color='silver', linestyle='--', linewidth=0.5,alpha=0.3)
ax.tick_params(labelsize=10,direction='in',top=True,left=True,bottom=True,right=True)
cbar.ax.tick_params(labelsize=10,which='major',direction='in')
cbar.ax.tick_params(which='minor',right=False)
ax.set_title(r'ML charge density (e/$\rm \AA^3$)',fontsize=12,pad=10)
fig.tight_layout()
fig.savefig('figures/'+f'ml_chg_map_test_{frame}.png',dpi=300,bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(6,6),dpi=100)
imshow_ax = ax.imshow(chg_diff_points.reshape((ngxf,ngyf),order='C'),cmap='seismic',origin='lower', vmin=vmin, vmax=vmax)
ax.set_xlim((0,ngxf-1))
ax.set_ylim((0,ngyf-1))
ax.set_xticks([tick for tick in range(0,ngxf,20)])
ax.set_yticks([tick for tick in range(0,ngyf,20)])
cbar = fig.colorbar(imshow_ax, ax=ax, extend='both',shrink=0.8)
cbar.minorticks_on()
ax.grid(color='silver', linestyle='--', linewidth=0.5,alpha=0.3)
ax.tick_params(labelsize=10,direction='in',top=True,left=True,bottom=True,right=True)
cbar.ax.tick_params(labelsize=10,which='major',direction='in')
cbar.ax.tick_params(which='minor',right=False)
ax.set_title(r'DFT$-$ML charge density (e/$\rm \AA^3$)',fontsize=12,pad=10)
fig.tight_layout()
fig.savefig('figures/'+f'diff_chg_map_test_{frame}.png',dpi=300,bbox_inches='tight')
plt.show()

test_r2 = metrics.r2_score(chg_dft_points,chg_pred_points)
test_mae = metrics.mean_absolute_error(chg_dft_points,chg_pred_points)
test_mse = metrics.mean_squared_error(chg_dft_points,chg_pred_points)
test_rmse = metrics.mean_squared_error(chg_dft_points,chg_pred_points,squared=False)
test_maxae = metrics.max_error(chg_dft_points,chg_pred_points)

print('           |        R2       |      RMSE       |       MAE       |      MaxAE      |       MSE     ',flush=True)
print(f'Test       |  {test_r2: 8.6e}  |  {test_rmse: 8.6e}  |  {test_mae: 8.6e}  |  {test_maxae: 8.6e}  |  {test_mse: 8.6e}  ',flush=True)

plot_corr_scatter(r'CHG (e/$\rm \AA^3$)',chg_dft_points,chg_pred_points,savefig=True,model_name=f'test_{frame}',show=True)