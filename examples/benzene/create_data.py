import numpy as np

import sys
sys.path.append('../../')
sys.path.append('../../jlgridfingerprints')
from jlgridfingerprints.fingerprints import JLGridFingerprints
from jlgridfingerprints.tools import create_grid_coords,sample_charge

from ase.io import read
from pymatgen.io.vasp.outputs import Chgcar
import h5py

from sklearn.utils import shuffle

from tqdm import tqdm
import time

import os

def write_chg_files(data, partition, frame=None):
    if not frame is None:
        h5f = h5py.File(f"y_{partition}_data_{frame}.h5", "w")
    else:
        h5f = h5py.File(f"y_{partition}_data.h5", "w")
    dset = h5f.create_dataset("chg", data=data, compression="gzip", compression_opts=9)
    h5f.close()
    del dset

def write_desc_files(data, partition, frame=None):

    if not frame is None:
        h5f = h5py.File(f"X_{partition}_data_{frame}.h5", "w")
    else:
        h5f = h5py.File(f"X_{partition}_data.h5", "w")

    dset = h5f.create_dataset("desc", data=data, compression="gzip", compression_opts=9)
    h5f.close()
    del dset

settings = {
    "rcut": 2.8,
    "nmax": [16,6],
    "lmax": 6,
    "alpha": [7,7],
    "beta": [0,0],
    "rmin": -0.78,
    "species": ["C", "H"],
    "body": "1+2",
    "periodic": False,
    'double_shifted': True,
}

jl = JLGridFingerprints(**settings)

print('Number of JL coefficients: ',jl._n_features)

sigma = 90.0
uniform_ratio = 0.30

scf_path = "scf_data"

num_frames = {"train": 3, "test": 1}
n_samples_per_snapshot = 1000

time_descriptor = 0.0
time_open = 0.0
time_write = 0.0

X_train_data = []
y_train_data = []
X_test_data = []
y_test_data = []
with tqdm(total=sum([num_frames[partition] for partition in ["train", "test"]])) as pbar:
    for partition in ["train", "test"]:
        for frame in range(num_frames[partition]):

            t_init = time.time()
            atoms = read(scf_path + f"/{partition}_data/{frame}/POSCAR")
            chgcar = Chgcar.from_file(scf_path + f"/{partition}_data/{frame}/CHGCAR")
            time_open += (time.time() - t_init)

            vol = atoms.get_volume()

            fcoords = create_grid_coords(grid_size=chgcar.data["total"].shape,return_cartesian_coords=False)
            chg_points = chgcar.data["total"].ravel() / vol

            selected_index = sample_charge(chg=chg_points,sigma=sigma,n_samples=n_samples_per_snapshot,uniform_ratio=uniform_ratio,seed=42)

            fcoords = fcoords[selected_index]
            cart_coords = np.dot(fcoords, atoms.get_cell().array)
            chg_points = chg_points[selected_index]

            t_init = time.time()
            jl_points = jl.create(atoms, cart_coords)
            time_descriptor += (time.time() - t_init)

            t_init = time.time()
            if partition == 'train':
                X_train_data.append(jl_points)
                y_train_data.append(chg_points)
            elif partition == 'test':
                X_test_data.append(jl_points)
                y_test_data.append(chg_points)
            time_write += (time.time() - t_init)

            pbar.update(1)

X_train_data = np.vstack(X_train_data)
y_train_data = np.hstack(y_train_data)

X_test_data = np.vstack(X_test_data)
y_test_data = np.hstack(y_test_data)

n_frames = sum([num_frames[partition] for partition in ["train", "test"]])
time_descriptor /= n_frames
time_open /= n_frames
time_write /= n_frames

print(f'JL coeff    : {time_descriptor:>5.3f} sec for {n_samples_per_snapshot} points over {n_frames} structures')
print(f'Open files  : {time_open:>5.3f} sec')
print(f'Write files : {time_write:>5.3f} sec')

X_train_data, y_train_data = shuffle(X_train_data, y_train_data, random_state=42)

write_desc_files(X_train_data, "train")
write_desc_files(X_test_data, "test")

write_chg_files(y_train_data, "train")
write_chg_files(y_test_data, "test")