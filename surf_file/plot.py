from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import cm
import h5py
import argparse
import numpy as np
from os.path import exists
import seaborn as sns
import sys 

def plot_2d_contour(surf_file, surf_name='train_loss', vmin=0.1, vmax=10, vlevel=0.5, threshold=10):
    """Plot 2D contour map and 3D surface."""

    f = h5py.File(surf_file, 'r')
    x = np.array(f['xcoordinates'][:])
    y = np.array(f['ycoordinates'][:])
    X, Y = np.meshgrid(x, y)
    Z = np.array(f[surf_name][:])

    Z = np.clip(Z,0,10)
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False, vmin=0, vmax=10)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.savefig(surf_file + '_' + surf_name + '_3dsurface.svg', dpi=300,
                bbox_inches='tight', format='svg')
    
    f.close()
    # plt.show()

path = ['adv_data_nips_data1_new3.h5', 'adv_data_nips_data10_newfinal.h5', 'adv_data_nips_data100_newfinal.h5', #2
        'adv_data_torch_data1_new3.h5', 'adv_data_torch_data10_newfinal.h5', 'adv_data_torch_data100_newfinal.h5', #5
        'adv_data_zico_data1_new3.h5', 'adv_data_zico_data10_newfinal.h5', 'adv_data_zico_data100_newfinal.h5', #8
        'clean_data_nips_data1_new3.h5', 'clean_data_nips_data10_newfinal.h5', 'clean_data_nips_data100_newfinal.h5', #11
        'clean_data_torch_data1_new3.h5', 'clean_data_torch_data10_newfinal.h5', 'clean_data_torch_data100_newfinal.h5', #14
        'clean_data_zico_data1_new3.h5', 'clean_data_zico_data10_newfinal.h5', 'clean_data_zico_data100_newfinal.h5'] #17

for pp in path:
    plot_2d_contour(pp)
