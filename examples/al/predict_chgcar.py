import numpy as np

import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import time

import pickle

from ase.io import read
from pymatgen.io.vasp.outputs import Chgcar

import sys
sys.path.append('../../')
sys.path.append('../../jlgridfingerprints')
from jlgridfingerprints.predictor import JLPredictor
from jlgridfingerprints.tools import create_grid_coords,sample_charge

from sklearn import metrics

def plot_corr_scatter(label, y_true, y_pred, savefig=False, save_path=".", model_name="", show=False):

    r2_score = metrics.r2_score(y_true, y_pred)
    mae_score = metrics.mean_absolute_error(y_true, y_pred)
    mse_score = metrics.mean_squared_error(y_true, y_pred)
    rmse_score = metrics.mean_squared_error(y_true, y_pred, squared=False)
    maxae_score = metrics.max_error(y_true, y_pred)

    min_ax = min(y_true.min(),y_pred.min())*0.9
    max_ax = max(y_true.max(),y_pred.max())*1.1 

    fig, ax = plt.subplots(figsize=(6, 6))

    x0, x1 = min_ax, max_ax
    lims = [max(x0, x0), min(x1, x1)]
    ax.plot(lims, lims, c="tab:blue", ls="--", zorder=1, lw=1)

    ax.scatter(y_true,y_pred,marker="x",color="tab:blue",linewidth=1.5,edgecolors=None,alpha=1.0)

    ax.set_xlabel("DFT " + label, fontsize=16)
    ax.set_ylabel("ML " + label, fontsize=16)

    ax.text(s=r"R$^2$ =" + f" {r2_score:.6f}",x=0.1 * (max_ax - min_ax) + min_ax,y=0.9 * (max_ax - min_ax) + min_ax,fontsize=16,ha="left",va="top")
    ax.text(s=r"RMSE  =" + f" {rmse_score:.6f}",x=0.5 * (max_ax - min_ax) + min_ax,y=0.1 * (max_ax - min_ax) + min_ax,fontsize=16,ha="left")
    ax.text(s=r"MAE   =" + f" {mae_score:.6f}",x=0.5 * (max_ax - min_ax) + min_ax,y=0.175 * (max_ax - min_ax) + min_ax,fontsize=16,ha="left")
    ax.text(s=r"MaxAE =" + f" {maxae_score:.6f}",x=0.5 * (max_ax - min_ax) + min_ax,y=0.25 * (max_ax - min_ax) + min_ax,fontsize=16,ha="left")
    
    ax.tick_params(labelsize=14, direction="in", top=True, right=True)

    ax.set_xlim((min_ax, max_ax))
    ax.set_ylim((min_ax, max_ax))

    ax.set_title(model_name, fontsize=14, ha="center")

    fig.tight_layout()

    if savefig:
        fig.savefig(save_path + "/" + model_name + "_scatter.png", dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    plt.close()

scf_path = "scf_data"
model_name = "linear_model_chg"
save_path = "chgcar_files"
nframes = 1
batch_size = 100000

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

jl = JLPredictor(jl_settings=settings, model_path=f"scikit_{model_name}.p", grid_size=(140,140,140))
# jl = JLPredictor(jl_settings=settings, model_path=f"scikit_{model_name}.p", encut=600.0,prec='Accurate')

if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists("poscar_files"):
    os.makedirs("poscar_files")
if not os.path.exists("figures"):
    os.makedirs("figures")

time_descriptor = 0.0
time_open = 0.0
time_write = 0.0

for frame in tqdm(range(nframes)):

    t_init = time.time()
    atoms = read(f"{scf_path}/test_data/{frame}/POSCAR")
    chgcar = Chgcar.from_file(f"{scf_path}/test_data/{frame}/CHGCAR")
    time_open += (time.time() - t_init)

    vol = atoms.cell.volume

    dft_chg = chgcar.data['total'].ravel() / vol

    t_init = time.time()
    ml_chg = jl.predict_chgcar(atoms,nelect=96,batch_size=batch_size,verbose=False,save_path=save_path,name=f'CHGCAR_test_{frame}',return_chg=True,write_chgcar=True)
    time_descriptor += (time.time() - t_init)

    ml_chg = ml_chg.ravel() / vol

    test_r2 = metrics.r2_score(dft_chg, ml_chg)
    test_mae = metrics.mean_absolute_error(dft_chg, ml_chg)
    test_mse = metrics.mean_squared_error(dft_chg, ml_chg)
    test_rmse = metrics.mean_squared_error(dft_chg, ml_chg, squared=False)
    test_maxae = metrics.max_error(dft_chg, ml_chg)

    print(f"           |        R2       |      RMSE       |       MAE       |      MaxAE      |       MSE     ",flush=True,)
    print(f" Test      |  {test_r2: 8.6e}  |  {test_rmse: 8.6e}  |  {test_mae: 8.6e}  |  {test_maxae: 8.6e}  |  {test_mse: 8.6e}  ",flush=True,)

    plot_corr_scatter(r"CHG (e/$\rm \AA^3$)",dft_chg,ml_chg,savefig=True,save_path="figures",model_name=f"test_{frame}",show=True)

    atoms.write(f"poscar_files/POSCAR_test_{frame}", vasp5=True, sort=True)

time_descriptor /= nframes
time_open /= nframes
time_write /= nframes

print(f'JL coeff    : {time_descriptor:>5.3f} sec for {nframes} structures')
print(f'Open files  : {time_open:>5.3f} sec')
print(f'Write files : {time_write:>5.3f} sec')