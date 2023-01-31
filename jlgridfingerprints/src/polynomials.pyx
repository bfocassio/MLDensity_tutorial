# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
import numpy as np
from cython.parallel import prange
cimport numpy as np
cimport cython

from lib.utils import vector_dot
from libc.math cimport pi,cos

def expand_jacobi(double[::1] rgi, int nmax, int alpha, int beta, double rcut, double rmin=0, double gamma=1, bint shifted=1, bint double_shifted=0):
    """Shifted Jacobi polynomial expansion for neighbour distances. See paper for the defination.

    Args:
        rij (array): Pair distances
        shifted (bool, optional): Use the shifted polynomial. Defaults to True.

    Returns:
        array : Jacobi expansion of shape (nmax,len(rij))
    """

    cdef int ndist = rgi.shape[0]
    cdef double[::1] dist = rgi
    cdef int deg_max = nmax + 1

    cdef int i = 0
    cdef int n = 0
    cdef double d = 0
    cdef double theta0 = 0

    cdef np.ndarray[dtype=double,ndim=1] cos_theta = np.empty(ndist,dtype=np.double)
    cdef double[::1] cos_theta0 = cos_theta

    cdef np.ndarray[dtype=double,ndim=2] pjacobi = np.empty((deg_max,ndist),dtype=np.double)
    cdef double[:,::1] vj = pjacobi

    for i in range(ndist):
        theta0 = pi * (dist[i] - rmin) / (rcut - rmin)
        cos_theta0[i] = gamma * cos(theta0)
    
    calculate_jacobi(nmax, alpha, beta, cos_theta0, gamma, shifted, double_shifted, vj)

    for i in range(ndist):
        if dist[i] > rcut:
            for n in range(1,deg_max):
                vj[n,i] = d

    if double_shifted:
        return pjacobi[2:,:]
    else:
        return pjacobi[1:,:]

def expand_legendre(int lmax, double[:,::1] hat_rgi, double[:,::1] hat_rgj, bint zero_diag=1):
    """Legendre polynomial expansion for scalar products. See paper for the defination.
        Args:
            hatrij (array): Array of unit vectors joining the neighbours.
            hatrik (array): Array of unit vectors joining the neighbours
            zero_diag (bool, optional): Zero the diagonal of the expansion matrix. Defaults to True.
        Returns:
            array : Legendre expansion of the scalar products for neighbours unit vectors.
        """

    cdef int num_n = hat_rgi.shape[0]
    cdef int num_m = hat_rgj.shape[0]
    cdef int deg_max = lmax + 1

    cdef int io = 0
    cdef int jo = 0

    cdef np.ndarray[dtype=double,ndim=2] rhatdot = np.empty((num_n,num_m),dtype=np.double)
    cdef double[:,::1] prod = rhatdot

    cdef np.ndarray[dtype=double,ndim=3] plegendre = np.empty((deg_max,num_n,num_m),dtype=np.double)
    cdef double[:,:,::1] vl = plegendre

    # maybe we could do this inside the calculate legendre so that we don't need the double loop twice
    # need testing
    vector_dot(hat_rgi, hat_rgj, prod)

    for io in range(num_n):
        for jo in range(num_m):
            if prod[io,jo] > 1.0: prod[io,jo] = 1.0
            elif prod[io,jo] < -1.0: prod[io,jo] = -1.0

    legendre_eval(lmax, rhatdot, zero_diag, vl)

    return plegendre

cdef void calculate_jacobi(int nmax,double alpha,double beta,double [::1]x, double gamma, bint shifted, bint double_shifted, double[:,::1] jac):

    cdef int i = 0
    cdef int deg = 1
    cdef int deg_max = nmax + 1
    cdef int ndist = x.shape[0]

    cdef np.ndarray[dtype=np.double_t,ndim=1] pjacobi0 = np.empty((deg_max),dtype=np.double)
    cdef double[::1] pj0 = pjacobi0

    cdef np.ndarray[dtype=np.double_t,ndim=1] pjacobi1 = np.empty((deg_max),dtype=np.double)
    cdef double[::1] pj1 = pjacobi1

    cdef double s = 0
    cdef double p1x = 0
    cdef double gfac = 0

    jacobi_eval(nmax, alpha, beta, x, jac)

    if shifted:
        jacobi_eval_single(nmax, alpha, beta,-1 * gamma, pj0)

        for i in range(ndist):
            for deg in range(1,deg_max):
                s = jac[deg,i] - pj0[deg]
                jac[deg,i] = s

        if double_shifted:
            jacobi_eval_single(nmax, alpha, beta, 1, pj1)

            for i in range(ndist):
                for deg in range(2,deg_max):
                    # s = jac[deg,i] - ((pj1[deg]-pj0[deg])/(pj1[1]-pj0[1])) * jac[1,i]
                    gfac = (gamma+x[i])/(gamma+1)
                    p1x = pj1[deg]-pj0[deg]
                    s = jac[deg,i] - gfac * p1x
                    jac[deg,i] = s


cdef void jacobi_eval_single(int nmax,double alpha, double beta, double x, double[::1] jac):
    
    cdef int deg_max = nmax + 1

    cdef double val = 1
    cdef double a = 0
    cdef double b = 0
    cdef double c = 0
    cdef double v1 = 0
    cdef double v2 = 0
    cdef int deg = 0

    jac[0] = val
    
    for deg in range(1,deg_max):
        a = deg + alpha
        b = deg + beta
        c = a + b
        if deg == 1:
            val = 0.5*(x-1.0)
            val *= c
            val += a
        else:
            v1 = -2.0*c * (a-1.0)*(b-1.0) * jac[deg-2]
            v2 = c*(c-2.0)*x + (a-b)*(c-2.0*deg)
            v2 *= (c-1.0)
            v2 *= jac[deg-1]
        
            v1 += v2
            v2 = 2.0*deg*(c-deg)*(c-2.0)
            val = v1/v2

        jac[deg] = val


cdef void jacobi_eval(int nmax,double alpha, double beta, double [::1]x, double[:,::1] jac):
    
    cdef int deg_max = nmax + 1

    cdef int i = 0
    cdef int ndist = x.shape[0]

    cdef double val = 1
    cdef double a = 0
    cdef double b = 0
    cdef double c = 0
    cdef double v1 = 0
    cdef double v2 = 0
    cdef double xi = 0
    cdef int deg = 0

    for i in range(ndist):
        jac[0,i] = val

    for i in range(ndist):
        val = 1
        v1 = 0
        v2 = 0
        xi = x[i]
        for deg in range(1,deg_max):
            a = deg + alpha
            b = deg + beta
            c = a + b
            if deg == 1:
                val = 0.5*(xi-1.0)
                val *= c
                val += a
            else:
                v1 = -2.0*c * (a-1.0)*(b-1.0) * jac[deg-2,i]
                v2 = c*(c-2.0)*xi + (a-b)*(c-2.0*deg)
                v2 *= (c-1.0)
                v2 *= jac[deg-1,i]
            
                v1 += v2
                v2 = 2.0*deg*(c-deg)*(c-2.0)
                val = v1/v2

            jac[deg,i] = val


cdef void legendre_eval(int lmax, double [:,::1]x, bint zero_diag, double[:,:,::1] jac):
    
    cdef int deg_max = lmax + 1

    cdef int i = 0
    cdef int j = 0
    cdef int idist = x.shape[0]
    cdef int jdist = x.shape[1]

    cdef double val = 1
    cdef double a = 0
    cdef double b = 0
    cdef double c = 0
    cdef double v1 = 0
    cdef double v2 = 0
    cdef double xij = 0
    cdef int deg = 0

    for i in range(idist):
        for j in range(jdist):
            jac[0,i,j] = val

    for i in range(idist):
        for j in range(jdist):
            if zero_diag and j <= i:
                continue
            val = 1
            v1 = 0
            v2 = 0
            xij = x[i,j]
            for deg in range(1,deg_max):
                a = deg
                b = deg
                c = a + b
                if deg == 1:
                    val = 0.5*(xij-1.0)
                    val *= c
                    val += a
                else:
                    v1 = -2.0*c * (a-1.0)*(b-1.0) * jac[deg-2,i,j]
                    v2 = c*(c-2.0)*xij + (a-b)*(c-2.0*deg)
                    v2 *= (c-1.0)
                    v2 *= jac[deg-1,i,j]
                
                    v1 += v2
                    v2 = 2.0*deg*(c-deg)*(c-2.0)
                    val = v1/v2

                jac[deg,i,j] = val

    if zero_diag:
        for i in range(idist):
            for j in range(jdist):
                for deg in range(deg_max):
                    if i == j:
                        jac[deg,i,j] = 0.0
                    elif j < i:
                        jac[deg,i,j] = jac[deg,j,i]
            