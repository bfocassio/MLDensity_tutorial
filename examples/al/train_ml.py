import numpy as np

import matplotlib.pyplot as plt
import time
import pickle

from sklearn.linear_model import Ridge
from sklearn import metrics
from sklearn.utils import shuffle

import h5py

import os

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
    
    ax.text(s=r'R$^2$ ='+f' {r2_score:.6f}',x=0.1*(max_ax-min_ax)+min_ax,y=0.9*(max_ax-min_ax)+min_ax,fontsize=16,ha='left',va='top')
    ax.text(s=r'RMSE  ='+f' {rmse_score:.6f}',x=0.5*(max_ax-min_ax)+min_ax,y=0.1*(max_ax-min_ax)+min_ax,fontsize=16,ha='left')
    ax.text(s=r'MAE   ='+f' {mae_score:.6f}',x=0.5*(max_ax-min_ax)+min_ax,y=0.175*(max_ax-min_ax)+min_ax,fontsize=16,ha='left')
    ax.text(s=r'MaxAE ='+f' {maxae_score:.6f}',x=0.5*(max_ax-min_ax)+min_ax,y=0.25*(max_ax-min_ax)+min_ax,fontsize=16,ha='left')
    #ax.text(s=r'MSE  ='+f' {mse_score: .6f}',x=0.5*(max_ax-min_ax)+min_ax,y=0.05*(max_ax-min_ax)+min_ax,fontsize=16,ha='left')

    ax.tick_params(labelsize=14,direction='in', top=True,right=True)

    ax.set_xlim((min_ax,max_ax))
    ax.set_ylim((min_ax,max_ax))

    ax.set_title(model_name,fontsize=14,ha='center')

    fig.tight_layout()
    
    if savefig:
        if not os.path.exists("figures"):
            os.makedirs("figures")
        fig.savefig('figures/'+model_name+'_scatter.png',dpi=300,bbox_inches='tight')

    if show:
        plt.show()
    plt.close()

def read_h5(partition):

    h5file = h5py.File(f'y_{partition}_data.h5', 'r')
    y_data = np.asarray(h5file['chg'][:])
    del h5file

    h5file = h5py.File(f'X_{partition}_data.h5', 'r')
    X_data = np.asarray(h5file['desc'][:])
    del h5file

    return X_data, y_data

X_train, y_train = read_h5('train')
X_test, y_test = read_h5('test')

print(X_train.shape)
print(y_train.shape)

X_train, y_train = shuffle(X_train, y_train, random_state=42)
print('Shuffled training set')

t_init = time.time()
NAME = f"linear_model_chg"
print(NAME,flush=True)

assert np.isfinite(X_train).all()
assert np.isfinite(X_test).all()

model = Ridge(alpha=0.0, fit_intercept=False, solver='svd')
model.fit(X_train, y_train)

assert np.isfinite(model.coef_).all()

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_r2 = metrics.r2_score(y_train,y_train_pred)
test_r2 = metrics.r2_score(y_test,y_test_pred)

train_mae = metrics.mean_absolute_error(y_train,y_train_pred)
test_mae = metrics.mean_absolute_error(y_test,y_test_pred)

train_mse = metrics.mean_squared_error(y_train,y_train_pred)
test_mse = metrics.mean_squared_error(y_test,y_test_pred)

train_rmse = metrics.mean_squared_error(y_train,y_train_pred,squared=False)
test_rmse = metrics.mean_squared_error(y_test,y_test_pred,squared=False)

train_maxae = metrics.max_error(y_train,y_train_pred)
test_maxae = metrics.max_error(y_test,y_test_pred)

print('           |        R2       |      RMSE       |       MAE       |      MaxAE      |       MSE     ',flush=True)
print(f'Training   |  {train_r2: 8.6e}  |  {train_rmse: 8.6e}  |  {train_mae: 8.6e}  |  {train_maxae: 8.6e}  |  {train_mse: 8.6e}  ',flush=True)
print(f'Test       |  {test_r2: 8.6e}  |  {test_rmse: 8.6e}  |  {test_mae: 8.6e}  |  {test_maxae: 8.6e}  |  {test_mse: 8.6e}  ',flush=True)

plot_corr_scatter(r'CHG (e/$\rm \AA^3$)',y_train,y_train_pred,savefig=True,model_name=NAME.replace('model_','model_train_'),show=True)
plot_corr_scatter(r'CHG (e/$\rm \AA^3$)',y_test,y_test_pred,savefig=True,model_name=NAME.replace('model_','model_test_'),show=True)

pickle.dump(model,open(f'scikit_{NAME}.p','wb'))

t_end = time.time()
m, s = divmod(t_end-t_init, 60)
h, m = divmod(m, 60)

print(f'Done in {int(h):d}h {int(m):02d}min {int(s):02d}sec\n',flush=True)