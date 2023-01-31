# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
import numpy as np
from cython.parallel import prange
cimport numpy as np
cimport cython

from sklearn.neighbors import KDTree

from libc.math cimport sqrt,ceil,abs

cdef inline double norm(double [::1]a):
    cdef double s = 0
    cdef double d = 0
    cdef double a1 = 0
    cdef int ndim = 3
    cdef int i = 0
    for i in range(ndim):
        a1 = a[i]
        s += a1*a1
    d = sqrt(s)
    return d

cdef inline double dot(double [::1]a,double [::1]b):
    cdef double s = 0
    cdef double d = 0
    cdef double a1 = 0
    cdef double b1 = 0
    cdef int ndim = 3
    cdef int i = 0
    for i in range(ndim):
        a1 = a[i]
        b1 = b[i]
        s += a1*b1
    return s

cdef inline void cross(double [::1]a,double [::1]b, double[::1] prod):

    cdef double d = 0
    cdef int i = 0
    cdef int n = 3
    
    for i in range(n):
        prod[i] = d

    prod[0] = a[1] * b[2] - a[2] * b[1]
    prod[1] = a[2] * b[0] - a[0] * b[2]
    prod[2] = a[0] * b[1] - a[1] * b[0]

cdef inline double get_volume(double[:,::1] cell):

    cdef double vol = 0

    vol  = cell[0,0] * (cell[1,1] * cell[2,2] - cell[1,2] * cell[2,1])
    vol += cell[0,1] * (cell[1,2] * cell[2,0] - cell[1,0] * cell[2,2])
    vol += cell[0,2] * (cell[1,0] * cell[2,1] - cell[1,1] * cell[2,0])

    return vol

cdef inline int get_num_reps(double[:,::1] cell, double vol, double rcut, int dir=0):

    cdef int i = 0
    cdef int j = 0
    cdef int k = 0
    if dir == 0:
        i = 0
        j = 1
        k = 2
    elif dir == 1:
        i = 1
        j = 2
        k = 0
    elif dir == 2:
        i = 2
        j = 0
        k = 1

    cdef double b1 = cell[j,1] * cell[k,2] - cell[j,2] * cell[k,1]
    cdef double b2 = cell[j,2] * cell[k,0] - cell[j,0] * cell[k,2]
    cdef double b3 = cell[j,0] * cell[k,1] - cell[j,1] * cell[k,0]

    cdef double coeff = vol / (b1*b1 + b2*b2 + b3*b3)
    
    cdef double p1 = b1 * coeff
    cdef double p2 = b2 * coeff
    cdef double p3 = b3 * coeff

    ni = <int>ceil(rcut / sqrt(p1*p1+p2*p2+p3*p3))
    
    return ni
    
cdef void get_extended_system(int nx, int ny, int nz, double [:,::1]positions, double [:,::1]cell, double[:,::1] ext_pos):

    cdef int natoms = positions.shape[0]
    cdef int i = 0, j = 0, k = 0, l = 0, u = 0

    l = 0
    for i in range(-nx,nx+1):
        for j in range(-ny,ny+1):
            for k in range(-nz,nz+1):
            
                for n in range(natoms):
                    for u in range(3):
                        ext_pos[l,u] = positions[n,u] + i * cell[0,u] + j * cell[1,u] + k * cell[2,u]
                    l = l + 1

def get_nn_in_sphere(double[:,::1] centers, double [:,::1]positions, double [:,::1]cell, long [::1]pbc, double radial_cutoff, int leaf_size=2):

    cdef int nx = 0, ny = 0, nz = 0
    cdef int natoms = positions.shape[0]
    cdef int ncenters = centers.shape[0]
    cdef double zero = 0
    
    cdef double vol = get_volume(cell)

    if pbc[0] == 1: nx = get_num_reps(cell,vol,radial_cutoff,0)
    if pbc[1] == 1: ny = get_num_reps(cell,vol,radial_cutoff,1)
    if pbc[2] == 1: nz = get_num_reps(cell,vol,radial_cutoff,2)

    cdef int nrep = (2*nx+1)*(2*ny+1)*(2*nz+1)
    
    cdef np.ndarray[dtype=double,ndim=2] ext_positions = np.empty((nrep*natoms,3),dtype=np.double)
    cdef double[:,::1] ext_pos = ext_positions

    get_extended_system(nx, ny, nz, positions, cell, ext_pos)
    nn_tree = KDTree(ext_positions, leaf_size=leaf_size)

    cdef np.ndarray[dtype=object,ndim=1] nn_list
    nn_list = nn_tree.query_radius(X=centers, r=radial_cutoff, return_distance=False)

    cdef object[::1] nn_list_view = nn_list
    cdef np.ndarray[dtype=int,ndim=1] num_nn_list = np.empty(ncenters,dtype=np.int32)
    cdef int[::1] num_nn_view = num_nn_list

    cdef int nntot = 0
    cdef int num_nn = 0
    cdef int ic = 0
    cdef int io = 0
    cdef int nn_index = 0
    cdef long[::1] center_nn

    for ic in range(ncenters):
        
        center_nn = nn_list_view[ic]
        num_nn = center_nn.shape[0]
        num_nn_view[ic] = num_nn
        nntot = nntot + num_nn

    cdef np.ndarray[dtype=int,ndim=1] nn_list_flat = np.empty(nntot,dtype=np.int32)
    cdef np.ndarray[dtype=double,ndim=1] nn_dist = np.empty(nntot,dtype=np.double)
    cdef np.ndarray[dtype=double,ndim=2] nn_vec = np.empty((nntot,3),dtype=np.double)
    cdef int[::1] flat_view = nn_list_flat
    cdef double[::1] dist_view = nn_dist
    cdef double[:,::1] vec_view = nn_vec

    cdef np.ndarray[dtype=int,ndim=1] start_stride = np.empty(ncenters,dtype=np.int32)
    cdef np.ndarray[dtype=int,ndim=1] end_stride = np.empty(ncenters,dtype=np.int32)
    cdef int[::1] init_stride = start_stride
    cdef int[::1] final_stride = end_stride

    cdef double vec_x = 0
    cdef double vec_y = 0
    cdef double vec_z = 0
    # cdef double thr = 1e-14
    cdef double dist = 0
    cdef double posx = 0, posy=0, posz=0
    cdef double cposx = 0, cposy=0, cposz=0
    cdef int flat_index = 0
    
    flat_index = 0
    for ic in range(ncenters):

        center_nn = nn_list_view[ic]
        num_nn = num_nn_view[ic]
        init_stride[ic] = flat_index

        cposx = centers[ic,0]
        cposy = centers[ic,1]
        cposz = centers[ic,2]

        for io in range(num_nn):
            
            nn_index = center_nn[io]
            flat_view[flat_index] = nn_index

            posx = ext_pos[nn_index,0]
            posy = ext_pos[nn_index,1]
            posz = ext_pos[nn_index,2]

            vec_x = posx - cposx
            vec_y = posy - cposy
            vec_z = posz - cposz
            
            dist = sqrt(vec_x*vec_x + vec_y*vec_y + vec_z*vec_z)
            # if dist < thr: dist = thr
            dist_view[flat_index] = dist
            vec_view[flat_index,0] = vec_x
            vec_view[flat_index,1] = vec_y
            vec_view[flat_index,2] = vec_z

            # if ic >= 845 and ic <= 852:
            #     print(ic)
            #     print(dist)
            #     print(vec_x,vec_y,vec_z)
            #     print(cposx,cposy,cposz)
            #     print(posx,posy,posz)
            #     print(flat_index)

            flat_index = flat_index + 1
        final_stride[ic] = flat_index

    return nn_list_flat,nn_dist,nn_vec,start_stride,end_stride,num_nn_list

        



