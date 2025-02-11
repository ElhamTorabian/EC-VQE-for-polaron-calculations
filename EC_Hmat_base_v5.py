
# %%
import numpy as np
from itertools import product
import more_itertools
import pandas as pd

# %%
def Hmatrix(E,t,w,g,N_site,n_ph):
    """
    This function generates the Hamiltonian matrix for the Holstein model.
    The Hamiltonian considered in this work can most generally be written as (with $\hbar = 1$):
    H=& \; \epsilon \displaystyle\sum_{i}^{N_s} a_i^{\dagger} a_i +\displaystyle\sum_{i,j>i}^{N_s} t_{ij} (a_i^{\dagger} a_j+a_j^{\dagger} a_i)\\+
        & \omega \displaystyle\sum_{i}^{N_s} b_{i}^{\dagger} b_{i}+ g \displaystyle\sum_{i=1}^{N_s} \; a_{i}^{\dagger} a_{i} (b_{i}^{\dagger} + b_{i}),
    where $a_i^{\dagger}$ creates a  bare particle and $b^\dagger_i$ creates a phonon on site $i$,  and $t_{ij}$ is the amplitude for the bare particle hopping between sites $i$ and $j$. 
    We consider the Holstein polaron model with an ordered one-dimensional lattice and $t_{ij} = t = 1$ restricted to nearest neighbour hopping.
    
    Parameters
    ----------
    E : float
        On-site energy.
    t : float
        Hopping parameter.
    w : float   
        Phonon frequency.
    g : float
        Electron-phonon coupling.
    N_site : int
        Number of sites.
    n_ph : int
        Number of phonons.

    Returns
    -------
    H : numpy array
        Hamiltonian matrix.
    """
    # Number of states
    d = N_site*(n_ph**N_site)

    # Generate all possible phonon states
    P = list(product(range(0, n_ph),repeat = N_site))
    Ph_states = [list(item) for item in P]

    # Generate all possible site states
    Site_states = []
    for i in range(N_site):
        a = np.zeros((1,N_site))
        a[0,i] = 1
        Site_states.append(list(a[0]))
    
    # Generate all possible states
    All = []
    for i in range(N_site):
        for j in range(len(Ph_states)):
            All.append(tuple([Site_states[i],Ph_states[j]]))

    # Generate Hamiltonian matrix
    
    t_shift = n_ph**(N_site) # Shift for the hopping term
    
    H = np.zeros((d, d))
    for i in range(d):
        # On-site energy and phonon energy
        H[i,i] = E+sum(All[i][1])*w

    for i in range(d-t_shift):
        # Hopping term
        H[i,i+t_shift] = t
        H[i+t_shift,i] = t

    I=0
    for i in range(N_site):
        g_shift = n_ph**(N_site-i-1) # Shift for the electron-phonon coupling term
        for j in range(n_ph**(i)):
            for  l in range(n_ph-1):
                for m in range(int(g_shift)):
                    A = g * np.sqrt(max(All[I][1][All[I][0].index(1)],All[(I)+g_shift][1][All[(I)+g_shift][0].index(1)]))
                    # Electron-phonon coupling term
                    H[I,(I)+g_shift] = A
                    H[(I)+g_shift,I] = A
                    I+=1
            I+=g_shift

    return(H)


# %%
def orthogonalize(U, eps=1e-20):
    """
    Orthogonalizes the matrix U (d x n) using Gram-Schmidt Orthogonalization.
    If the columns of U are linearly dependent with rank(U) = r, the last n-r columns
    will be 0.

    Args:
        U (numpy.array): A d x n matrix with columns that need to be orthogonalized.
        eps (float): Threshold value below which numbers are regarded as 0 (default=1e-15).

    Returns:
        (numpy.array): A d x n orthogonal matrix. If the input matrix U's cols were
            not linearly independent, then the last n-r cols are zeros.

    """
    Q, R = np.linalg.qr(U)
    return Q


# %%
def transformed_V(V_low,range_V,N_site,n_ph):
    """
    This function transforms the input vector V_low (with lower dimension) to a vector V with the original dimensions.
    
    Parameters
    ----------
    V_low : numpy array
        Input vector.
    range_V : list
        List of the range of the states.
    N_site : int
        Number of sites.
    n_ph : int
        Number of phonons.

    Returns
    -------
    V : numpy array
        Transformed vector.
    """
    # Number of states
    d = N_site*(n_ph**N_site)

    
    V = np.zeros((1,d))
    In = range_V[0]*(n_ph**N_site)
    m = 0
    for i in range(int(len(V_low)/(n_ph**len(range_V)))):
        n = 0
        for j in range(int(n_ph**len(range_V))):
            # shift the values of the vector V_low to the original vector V
            V[0][In+n] = V_low[m]
            n += n_ph**(N_site-range_V[0]-len(range_V))
            m += 1
        In += n_ph**N_site
    return(V)


# %%
def Norm(train_points):
    """
    This function calculates the norm of the training matrix.

    Parameters
    ----------
    train_points : numpy array
        Input matrix.

    Returns
    -------
        Norm of the training matrix
    """
    
    return(np.dot(train_points.transpose(),train_points))

# %%
def EC_result(Ch,N_site,n_p,E,t,w,g):
    """
    This function calculates the ground state energy of the Holstein model using the Eigenvector Continuation method.

    Parameters
    ----------
    Ch : int
        Number of sites in each chunk.
    type : str
        Type of the training points.
    N_site : int
        Number of sites.
    n_p : int   
        Number of phonons.
    E : float   
        On-site energy.
    t : float
        Hopping parameter.
    w : float
        Phonon frequency.
    g : float   
        Electron-phonon coupling.

    Returns
    -------
    min(np.real(eigenvalues)) : float
        Ground state energy.
    """
    M = more_itertools.windowed(range(0,N_site), Ch)
    E_total = []
    V_total = []

    for c in M:
        print("The subsystem is:\n",c)
        Hn = Hmatrix(E=E,t=t,w=w,g=g,N_site=len(c),n_ph=n_p)
        En,Vn = np.linalg.eig(Hn)
        print("Energy of subsystem is:   ", min(En))
        # print("Eigenvector for the small chunk:\n",np.real(Vn[:, np.argmin(En)]))
        V_transformed = transformed_V(np.real(Vn[:, np.argmin(En)]),c,N_site,n_p)
        # print("Eigenvector after transformation:\n",V_transformed)
        E_total.append(min((En)))
        V_total.append(V_transformed.reshape(-1,1))

    
    V_t = np.column_stack(V_total)
    V_tt = orthogonalize(V_t,eps=1e-20)
    norm = Norm(V_tt)
    # print("Overlap matrix is:\n", norm)


    P = matp(V_tt,E=E,t=t,w=w,g=g,N_site=N_site,n_ph=n_p)

    H_hat = np.matmul(V_tt.T , P)
    print("H_hat is:\n", H_hat)


    eigenvalues, eigenvectors = np.linalg.eig(H_hat)
    smallest_index = np.argmin(eigenvalues)
    smallest_eigenvector = eigenvectors[:, smallest_index]

    return (min(np.real(eigenvalues)), np.matmul(V_t,smallest_eigenvector))


N_sites = 6
N_p = 2

Energy, eigenvector = EC_result(Ch = 2, N_site=N_sites,n_p=N_p,E=0,t=1,w=0.1,g=0.449672447615957)

print("EC Ground state energy is: ", Energy)

# %%
H = Hmatrix(E=0,t=1,w=0.1,g=0.449672447615957,N_site=6,n_ph=2)
E,V = np.linalg.eig(H)
print("Exact Ground state energy is: ", min(np.real(E)))


# %%



