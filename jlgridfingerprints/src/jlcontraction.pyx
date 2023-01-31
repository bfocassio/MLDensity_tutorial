# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
import numpy as np
from cython.parallel import prange
cimport numpy as np
cimport cython
from libc.stdlib cimport malloc,free

def calculate_2b(double [:,::1]a):

    cdef int nmax = a.shape[0]
    cdef int num_i = a.shape[1]
    cdef int i = 0
    cdef int n = 0
    cdef double d = 0

    cdef double s = 0

    cdef np.ndarray[dtype=double,ndim=1] jacout = np.empty(nmax,dtype=np.double)
    cdef double[::1] jac = jacout

    for n in range(nmax):
        jac[n] = d

    for n in range(nmax):
        s = 0
        for i in range(num_i):
            s = s + a[n,i]
        jac[n] = s

    return jacout

def calculate_3b(double [:,::1]a,double [:,::1]b,double [:,:,::1]c):
    cdef int nmax = a.shape[0]
    
    cdef int lmax = c.shape[0]

    cdef int n4 = a.shape[1]
    cdef int n5 = b.shape[1]
    
    cdef int ndes = nmax*nmax*(lmax)
    cdef np.ndarray[dtype=double,ndim=1] jacout = np.empty(ndes,dtype=np.double)

    cdef double[::1] jac = jacout

    cdef int i1,i2,i3,i4,i5,i
    cdef double d = 0

    cdef double s = 0
    cdef double s1 = 0
    cdef double a1 = 0
    cdef double b1 = 0
    cdef double c1 = 0

    for i in prange(ndes,nogil=True):
        jac[i] = d

    cdef int k1 = nmax*lmax

    for i1 in prange(nmax,nogil=True):
        for i2 in prange(nmax):
            for i3 in prange(lmax):
                i = i1*k1 + i2*lmax + i3
                s = 0
                for i4 in range(n4):
                    a1 = a[i1,i4]
                    s1 = 0
                    for i5 in range(n5):
                        b1 = b[i2,i5]
                        c1 = c[i3,i4,i5]
                        s1 = s1 + b1*c1
                    s = s + s1*a1
                jac[i] = s
                        
    return jacout

def calculate_3b_upper(double [:,::1]a,double [:,::1]b,double [:,:,::1]c):
    cdef int nmax = a.shape[0]
    cdef int lmax = c.shape[0]
    
    cdef int n4 = a.shape[1]
    cdef int n5 = b.shape[1]

    cdef int ndes = (nmax*(nmax+1)*(lmax)/2)

    cdef np.ndarray[dtype=double,ndim=1] jacout = np.empty(ndes,dtype=np.double)

    cdef double[::1] jac = jacout

    cdef int i1,i2,i3,i4,i5,i
    cdef double d = 0.0

    cdef double s = 0
    cdef double s1 = 0
    cdef double a1 = 0
    cdef double b1 = 0
    cdef double c1 = 0

    for i in prange(ndes,nogil=True):
        jac[i] = d

    cdef int k1 = (nmax*(nmax+1)/2)

    for i3 in prange(lmax,nogil=True):
        for i1 in prange(nmax):
            for i2 in prange(nmax):
                if (i1 >= i2):
                    i = <int>(i1*nmax - (i1*(i1-1)/2) + (i2-i1) + i3*k1)
                    s = 0
                    for i4 in range(n4):
                        a1 = a[i1,i4]
                        s1 = 0
                        for i5 in range(n5):
                            b1 = b[i2,i5]
                            c1 = c[i3,i4,i5]
                            s1 = s1 + (b1*c1)
                        s = s + s1*a1
                    jac[i] = s
                        
    return jacout

cdef int compare_two_4b_index(int[:] point1,int[:] point2):
    cdef int idx,idy,idz,m1,m2
    cdef int a = 0

    for idx in range(3):
        m1 = 0
        m2 = 0

        for idy in range(3):
            idz = (idy + idx)%3
            if (point1[idz] == point2[idy]) and (point1[idz+3] == point2[idy+3]):
                m1 += 1
            idz = 2 - idz
            if (point1[idz] == point2[idy]) and (point1[idz+3] == point2[idy+3]):
                m2 += 1
        
        if (m1 == 3) or (m2 == 3):
            a = 1
    return a

cdef int compare_list_4b(int[:,::1] plist, int npoints):
    cdef int m = 0
    cdef int i,j

    cdef np.ndarray[dtype=int,ndim=1]  mp1 = np.empty(6,dtype=np.int32)
    cdef int[:] vmp1 = mp1
    cdef np.ndarray[dtype=int,ndim=1]  mp2 = np.empty(6,dtype=np.int32)
    cdef int[:] vmp2 = mp2
        

    if npoints == 0:
        m = 0
    else:
        
        for j in range(6):
            vmp2[j] = plist[npoints,j]
            
        for i in range(npoints):
            for j in range(6):
                vmp1[j] = plist[i,j]
            m = compare_two_4b_index(vmp1,vmp2)
            if m == 1:
                break

    return m

def get_3b_index(int nmax,int lmax,int multi_species = 0):
    
    if multi_species == 0:
        ndes = nmax*(nmax+1)*(lmax+1)/2
    else:
        ndes = nmax*nmax*(lmax+1)



    cdef np.ndarray[dtype=int,ndim=2]  des = np.empty((ndes,3),dtype=np.int32)
    cdef int[:,::1] vdes = des

    cdef int npoints = 0
    newp = 1
    for n1 in range(nmax):
        for n2 in range(nmax):
            for l1 in range(lmax+1):
                if (multi_species == 0) and (n2 < n1):
                    newp = 0
                # elif multi_species == 1: accept all point no condition necessary
                    
                if newp == 1:
                    vdes[npoints,0] = n1
                    vdes[npoints,1] = n2
                    vdes[npoints,2] = l1
                    npoints += 1

                else:
                    newp = 1
                    
    return des

def get_4b_index(int nmax,int lmax,int multi_species = 0):
    cdef int n1,n2,n3,l1,l2,l3

    cdef ndes = nmax*nmax*nmax*(lmax+1)*(lmax+1)*(lmax+1)

    cdef np.ndarray[dtype=int,ndim=2]  des = np.empty((ndes,6),dtype=np.int32)
    cdef int[:,::1] vdes = des

    cdef int match

    cdef int npoints = 0
    cdef int j = 0
    for n1 in range(nmax):
        for l1 in range(lmax+1):
            for n2 in range(nmax):
                for l2 in range(lmax+1):
                    for n3 in range(nmax):
                        for l3 in range(lmax+1):

                            vdes[npoints,0] = n1
                            vdes[npoints,1] = n2
                            vdes[npoints,2] = n3
                            vdes[npoints,3] = l1
                            vdes[npoints,4] = l2
                            vdes[npoints,5] = l3

                            match = compare_list_4b(vdes,npoints)
                            if match != 1:
                                npoints += 1
    
    cdef np.ndarray[dtype=int,ndim=2]  rdes = np.empty((npoints,6),dtype=np.int32)
    cdef int[:,:] vrdes = rdes
    cdef int k
    for k in range(npoints):
        for j in range(6):
            vrdes[k,j] = vdes[k,j]
    return rdes 

def calculate_4b(double [:,:]a,double [:,:]b,double [:,:]c,double [:,:,:]d,double [:,:,:]e,double [:,:,:]f,int[:,:] index,double [:] jac):
    cdef int nmax = a.shape[0]
    cdef int lmax = d.shape[0]
    
    cdef int n7 = a.shape[1]
    cdef int n8 = b.shape[1]
    cdef int n9 = c.shape[1]

    

    cdef int ndes = index.shape[0]


    #cdef np.ndarray[dtype=double,ndim=1] jacout = np.empty(ndes,dtype=np.double)

    #cdef double[::1] jac = jacout


    cdef int i1,i2,i3,i4,i5,i6,i7,i8,i9,i

    for i in range(ndes):
        i1 = index[i,0]
        i2 = index[i,1]
        i3 = index[i,2]
        i4 = index[i,3]
        i5 = index[i,4]
        i6 = index[i,5]
        jac[i] = 0.0
        for i7 in range(n7):
            for i8 in range(n8):
                for i9 in range(n9):
                    jac[i] += (a[i1,i7]*b[i2,i8]*c[i3,i9]*d[i4,i7,i8]*e[i5,i8,i9]*f[i6,i9,i7])
    # return jacout

def calculate_3b_by_index(double [:,:]a,double [:,:]b,double [:,:,:]c,int[:,:] index,double [:] jac):
    cdef int nmax = a.shape[0]
    cdef int lmax = c.shape[0]

    cdef int n4 = a.shape[1]
    cdef int n5 = b.shape[1]
    
    cdef int ndes = index.shape[0]

    cdef int i1,i2,i3,i

    #cdef np.ndarray[dtype=double,ndim=1] jacout = np.empty(ndes,dtype=np.double)
    #cdef double[::1] jac = jacout
    
    for i in range(ndes):
        i1 = index[i,0]
        i2 = index[i,1]
        i3 = index[i,2]
        jac[i] = 0.0
        for i4 in range(n4):
            for i5 in range(n5):
                jac[i] += (a[i1,i4]*b[i2,i5]*c[i3,i4,i5])
                        
    # return jacout