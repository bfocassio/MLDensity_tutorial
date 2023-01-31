# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
import numpy as np
from cython.parallel import prange
cimport numpy as np
cimport cython

from libc.math cimport sqrt,pi

def get_versors(double [:,::1]vector,double [::1]distance):

    cdef int num_n = distance.shape[0]
    cdef int dim = 3

    cdef np.ndarray[dtype=np.double_t,ndim=2] versor = np.empty((num_n,dim),dtype=np.double)
    cdef double[:,::1] vhat = versor

    _get_versors(vector,distance,vhat)

    return versor

def vector_norm(double [:,::1]vector):

    cdef int num_n = vector.shape[0]
    cdef np.ndarray[dtype=double,ndim=1] dist = np.empty(num_n,dtype=np.double)
    cdef double[::1] norm = dist

    _vector_norm(vector, norm)
    
    return dist

def vector_from_point( double[::1]point1, double[:,::1]list_of_points):

    cdef int ndim = 3
    cdef int npoints = list_of_points.shape[0]
    cdef np.ndarray[dtype=double,ndim=2] dist_vector = np.empty((npoints,ndim),dtype=np.double)
    cdef double[:,::1] vec = dist_vector

    _vector_from_point(point1,list_of_points,vec)

    return dist_vector

def vector_dot(double[:,::1] vector_a,double[:,::1] vector_b, double [:,::1] prod):
    return _vector_dot(vector_a,vector_b,prod)

cdef void _get_versors(double [:,::1]vector,double [::1]distance, double[:,::1] vhat):

    cdef int num_n = distance.shape[0]
    cdef int n = 0
    cdef int j = 0
    cdef int dim = 3
    cdef double d = 0
    cdef double s = 0
    cdef double dist = 0
    cdef double thr = 1e-4

    with nogil:
        for n in prange(num_n):
            dist = distance[n]
            if dist > thr:
                for j in range(dim):
                    vhat[n,j] = vector[n,j]/dist
            else:
                for j in range(dim):
                    vhat[n,j] = d
            # for j in range(dim):
            #     vhat[n,j] = vector[n,j]/dist

cdef void _vector_norm(double [:,::1]vector, double[::1] norm):

    cdef int num_n = vector.shape[0]

    cdef int i = 0
    cdef int n = 0
    cdef double d = 0
    cdef double s = 0
    cdef double s1 = 0
    cdef double a = 0
    cdef int ndim = 3
    cdef int l = 0
    cdef double thr = 1e-4

    for i in range(num_n):
        norm[i] = d

    for n in range(num_n):
        s1 = 0
        s = 0
        for l in range(ndim):
            a = vector[n,l]
            s1 = s1 + a * a
        s = sqrt(s1)
        if s > thr:
            norm[n] = s

cdef void _vector_from_point( double[::1]point1, double[:,::1]list_of_points, double[:,::1] vec):

    cdef int ndim = 3
    cdef int u = 0
    cdef double d = 0
    cdef int npoints = list_of_points.shape[0]

    cdef double[::1] ref_point = point1
    cdef double[:,::1] end_points = list_of_points

    cdef double s = 0

    for u in range(ndim):
        for i in range(npoints):
            vec[i,u] = d

    for i in range(npoints):
        for u in range(ndim):
            vec[i,u] = end_points[i,u] - ref_point[u]


cdef void _vector_dot(double[:,::1] vector_a,double[:,::1] vector_b, double [:,::1] prod):

    cdef int num_n = vector_a.shape[0]
    cdef int num_m = vector_b.shape[0]

    cdef int i = 0
    cdef int j = 0
    cdef int n = 0
    cdef int m = 0
    cdef double d = 0
    cdef double s = 0
    cdef int l = 0
    cdef int ndim = 3
    
    for i in prange(num_n,nogil=True):
        for j in prange(num_m):
            prod[i,j] = d

    for n in prange(num_n,nogil=True):
        for m in prange(num_m):
            s = 0
            for l in range(ndim):
                s = s + vector_a[n,l] * vector_b[m,l]
            prod[n,m] = s