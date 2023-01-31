import numpy as np
import time
from pymatgen.io.vasp.outputs import Chgcar
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.ase import AseAtomsAdaptor

from jlgridfingerprints.fingerprints import JLGridFingerprints

import json
import pickle
import os

from ase.units import Bohr, Rydberg

class JLPredictor():

    def __init__(self,jl_settings,model_path,grid_size=None,encut=None,prec='Accurate',scaler_path=None):

        if type(jl_settings) == str and jl_settings.endswith('.json'):
            self.jl_settings = json.load(open(jl_settings,'r'))
        elif type(jl_settings) == dict: 
            self.jl_settings = jl_settings
        else:
            raise ValueError('jl_settings argument needs to be a dict or path to json file')

        print('Using jl_settings: ',self.jl_settings,flush=True)

        self.jl = JLGridFingerprints(**self.jl_settings)
        print('Number of JL coefficients: ',self.jl._n_features,flush=True)

        print('Loading model from: ',model_path,flush=True)
        self.model = pickle.load(open(model_path, "rb"))

        if not scaler_path is None:
            self.scaler = pickle.load(open(scaler_path, "rb"))
        
        self.grid_size = grid_size

        self._encut = encut
        # I need to check the proper multiplication for encut with respect to prec

        if prec.lower().startswith('a'):
            self._prec_factor = 2.0
        elif prec.lower().startswith('n'):
            self._prec_factor = 2.0
        else:
            self._prec_factor = 1.0

    def predict_chgcar(self,atoms,nelect,batch_size=None,verbose=False,save_path=None,name=None,return_chg=False,write_chgcar=True,use_scaler=False):

        time_descriptor = 0.0
        time_write = 0.0

        vol = atoms.cell.volume

        if not self.grid_size is None and len(self.grid_size) == 3:
            ngxf, ngyf, ngzf = np.array(self.grid_size,dtype=int)
        else:
            alats = np.linalg.norm(atoms.get_cell().array,axis=-1)
            ngxf, ngyf, ngzf = self.get_chgcar_grid(alats,self._encut,self._prec_factor)

        if verbose: print(f'Grid size is: {ngxf}x{ngyf}x{ngzf}',flush=True)

        xx,yy,zz = np.meshgrid(np.arange(0,1,1/ngxf),np.arange(0,1,1/ngyf),np.arange(0,1,1/ngzf),indexing='ij')

        frac_points = np.vstack([xx.ravel(),yy.ravel(),zz.ravel()]).T
        cart_positions = np.dot(frac_points, atoms.get_cell().array)
        del frac_points

        t_init = time.time()
        if batch_size is None:
            X_batch = self.jl.create(atoms, cart_positions)
            if use_scaler:
                X_batch = self.scaler.transform(X_batch)
            ml_chg_points = self.model.predict(X_batch)
        else:
            nchunks = int(np.ceil(len(cart_positions) / batch_size))
            for ib in range(nchunks):
                X_batch = self.jl.create(atoms, cart_positions[ib*batch_size:(ib+1)*batch_size])
                if use_scaler:
                    X_batch = self.scaler.transform(X_batch)
                if ib == 0:
                    ml_chg_points = self.model.predict(X_batch)
                else: ml_chg_points = np.append(ml_chg_points,self.model.predict(X_batch),axis=0)
            del X_batch
        time_descriptor += (time.time() - t_init)

        ml_chg_points = ml_chg_points.reshape((ngxf,ngyf,ngzf),order='C') * vol
        ml_nelect = ml_chg_points.sum() / ml_chg_points.size
        if abs(ml_nelect-nelect) > 1e-6:
            ml_chg_points = self.normalize_nelect(ml_chg_points,nelect=nelect,volume=vol)

        if write_chgcar:
            t_init = time.time()
            chgcar = Chgcar(poscar=Poscar(AseAtomsAdaptor.get_structure(atoms)), data={'total': ml_chg_points}, data_aug=None)
            chgcar.data_aug["total"] = []

            if save_path is None:
                save_path = ''
            else:
                if not save_path.endswith('/'):
                    save_path += '/'
            if name is None:
                chgcar.write_file(save_path+'CHGCAR')
            else:
                chgcar.write_file(save_path+name)

            time_write += (time.time() - t_init)

        if verbose:
            print(f'JL coeff    : {time_descriptor:>5.3f} sec for {ngxf*ngyf*ngzf} points ({ngxf}x{ngyf}x{ngzf} grid) ',flush=True)
            if write_chgcar: print(f'Write files : {time_write:>5.3f} sec',flush=True)

        if return_chg:
            return ml_chg_points

    def get_chgcar_grid(self,alats,encut,prec_factor,wfact=4):

        def fftchk(grid):

            def fftchk_legal(nin):
                ifact = [2,3,5,7]
                n2div = 0
                n = nin
                for fact in ifact:
                    while n%fact == 0:
                        n = n / fact
                        if fact == 2:
                            n2div += 1
                if n == 1 and n2div!=0:
                    return True
                else:
                    return False

            for i in range(3):
                while not fftchk_legal(grid[i]):
                    grid[i] += 1

            return grid

        ngx,ngy,ngz = np.floor((encut/Rydberg)**0.5 / (2*np.pi/(alats/Bohr)) * wfact + 0.5).astype(int)
        ngx,ngy,ngz = fftchk([ngx,ngy,ngz])
        ngxf,ngyf,ngzf = prec_factor * np.asarray([ngx,ngy,ngz])

        return int(ngxf),int(ngyf),int(ngzf)

    def normalize_nelect(self,chg,nelect,volume):

        from scipy import fft

        ngxf,ngyf,ngzf = chg.shape

        chg /= volume

        chg_g = fft.fftn(chg)
        chg_g[0,0,0] = nelect / volume * (ngxf*ngyf*ngzf)
        chg_r = fft.ifftn(chg_g).real

        chg_r *= volume
          
        return chg_r