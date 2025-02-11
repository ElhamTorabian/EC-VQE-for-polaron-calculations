# %%
from qiskit.quantum_info import SparsePauliOp
import numpy as np
from qiskit.circuit.library import TwoLocal
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_algorithms import VQE
from qiskit_symb.quantum_info import Statevector
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
def matp(b,E,t,w,g,N_site,n_ph):
    """
    This function generates the matrix product of the Hamiltonian matrix and the training matrix b.

    Parameters
    ----------
    b : numpy array
        Input matrix.
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
    c2 : numpy array
        Matrix product of the Hamiltonian matrix and the training matrix b.
    """

    # Number of states
    d = N_site*(n_ph**N_site)


    ar,ac = tuple([len(b[:,0]),len(b[:,0])]) # n_rows and n_cols of training matrix

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


    br,bc = b.shape # n_rows * n_cols of training matrix

    t_shift = n_ph**(N_site) # Shift for the hopping term
    
    # Generate matrix product
    c2 = np.zeros((ar, bc))
    for i in range(d):
        for k in range(bc):
            # On-site energy and phonon energy
            A = E+sum(All[i][1])*w
            c2[i,k] += A * b[i,k]


    for i in range(d-t_shift):
        for k in range(bc):
            # Hopping term
            C = t
            c2[i,k] += C * b[i+t_shift,k]
            c2[i+t_shift,k] += C * b[i,k]


    for k in range(bc):
        I=0
        for i in range(N_site):
            g_shift = n_ph**(N_site-i-1) # Shift for the electron-phonon coupling term
            for j in range(n_ph**(i)):
                for  l in range(n_ph-1):
                    for m in range(int(g_shift)):
                        # Electron-phonon coupling term
                        B = g * np.sqrt(max(All[I][1][All[I][0].index(1)],All[(I)+g_shift][1][All[(I)+g_shift][0].index(1)]))
                        c2[(I),k] += B * b[(I)+g_shift,k]
                        c2[(I)+g_shift,k] += B * b[(I),k]
                        I+=1
                I+=g_shift
    return c2

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
def MyVQE(H_op,iter,k=1):
    """
    This function calculates the VQE for a given Hamiltonian matrix.

    Parameters
    ----------
    H_op : SparsePauliOp
        Hamiltonian matrix in terms of Pauli operators.
    iter : int
        Number of iterations.
    k : int
        Number of quantum circuit layer repetitions.

    Returns
    -------
    result.eigenvalue.real : float
        GS energy.
    psi : numpy array
        GS eigenvector.
    """
    # Build the ansatz
    ansatz = TwoLocal(H_op.num_qubits, 'ry', 'cx', 'linear', reps=k, insert_barriers=True)
    
    # Set the optimizer
    spsa = SPSA(maxiter=iter)

    counts = []
    values = []
    def store_intermediate_result(eval_count, parameters, mean, std):
        counts.append(eval_count)
        values.append(mean)

    seed = 170
    algorithm_globals.random_seed = seed

    # Set the noiseless simulator
    noiseless_estimator = AerEstimator(
        run_options={"seed": seed, "shots": 1024},
        transpile_options={"seed_transpiler": seed},
    )

    # Run the VQE
    vqe = VQE(noiseless_estimator, ansatz, optimizer=spsa, callback=store_intermediate_result)
    result = vqe.compute_minimum_eigenvalue(operator=H_op)

    print("optimal_circuit is:\n", result.optimal_circuit)
    state_vec = Statevector(result.optimal_circuit.decompose())
    sv_func = state_vec.to_lambda()

    params = result.optimizer_result.x
    psi = sv_func(*params)

    return(result.eigenvalue.real,psi)

H1 = np.array([[1, 0.1, 0, 0],
             [0.1, 1,0.1 , 0],
             [0, 0.1, 1, 0.1],
             [0, 0, 0.1, 1]])
H_op = SparsePauliOp.from_operator(H1)
E_test,V_test = MyVQE(H_op,100,1)
print("VQE result is:\n", E_test)
print(V_test)

# %%
def EC_result(Ch,type,N_site,n_p,E,t,w,g,k):
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

    print("---------------------------------")
    print("g is:  ",g)
    if type == "regular":
        # Divide the system into equal chunks
        M = more_itertools.chunked(range(0,2*Ch), Ch, strict=False)
        a = more_itertools.chunked(range(0,N_site), Ch)
        d_matp = int(len(list(a))) # Dimention of effective Hamiltonian
        
    elif type == "overlap":
        # Divide the system into overlapping chunks
        M = more_itertools.windowed(range(0,2*Ch), Ch)
        a = more_itertools.windowed(range(0,N_site), Ch)
        d_matp = int(len(list(a))) # Dimention of effective Hamiltonian
    
    # Training points GS energy and eigenvectors
    E_total = []
    V_total = []

    # Effective Hamiltonian
    H_hat =np.zeros((d_matp,d_matp))

    # Calculate the GS energy of each chunk
    Hn = Hmatrix(E=E,t=t,w=w,g=g,N_site=Ch,n_ph=n_p)

    H_op = SparsePauliOp.from_operator(Hn) # Hamiltonian matrix in terms of Pauli operators
    N_iter = 700
    En,Vn = MyVQE(H_op,N_iter,k) # VQE calculation
    print("One chunk of %s sites subsystem energy is:  %s \n"%(Ch,En))
    
    for c in M:
        # Transform the eigenvector to the original dimension
        V_transformed = transformed_V(np.real(Vn),c,2*len(c),n_p)
        E_total.append(En)
        V_total.append(V_transformed.reshape(-1,1))

    # Orthogonalize the training points
    V_t = np.column_stack(V_total)
    V_tt = orthogonalize(V_t)
    norm = Norm(V_tt) # To check if the training points are orthogonalized

    # Calculate the matrix product of the Hamiltonian matrix and the training matrix
    P = matp(V_tt,E=E,t=t,w=w,g=g,N_site=2*len(c),n_ph=n_p)
    H_hat4 = np.matmul(V_tt.T , P) # Effective Hamiltonian for the system with 4 sites

    # Leveraging the symmetry of the Hamiltonian to generate the effective Hamiltonian for the whole system
    A = H_hat4.copy()
    H_hat4_rep = A
    N_TP = len(E_total)
    L = int(N_TP/2)
    H_hat4_rep[0][L:] = -A[0][L:]
    for i in range(L):
        i+=L
        H_hat4_rep[i][0] = -A[i][0]
    l = 0
    H_hat[l:l+N_TP,l:l+N_TP]= H_hat4
    
    for i in range(int((H_hat.shape[0]/L)-N_TP)):
        l+=int(N_TP/2)
        H_hat[l:l+N_TP,l:l+N_TP]= H_hat4_rep

    # Calculate the GS energy of the effective Hamiltonian
    eigenvalues, eigenvectors = np.linalg.eig(H_hat)

    smallest_index = np.argmin(eigenvalues)
    smallest_eigenvector = eigenvectors[:, smallest_index]

    return (min(np.real(eigenvalues)))

print("The EC result is:  ", EC_result(2,"overlap",100,2,0,1,0.1,1,2))

# %%



