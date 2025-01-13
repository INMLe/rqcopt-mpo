from functools import reduce

from jax.numpy import eye, kron, asarray
from jax import config as c
c.update("jax_enable_x64", True)

from .fermionic_systems import get_swap_network_trotter_gates_fermi_hubbard_1d, get_swap_network_trotter_gates_molecular
from .spin_systems import get_brickwall_trotter_gates_spin_chain
from .util import get_identity_layers

# Define some operators
I = eye(2)
X = asarray([[0., 1.],[1., 0.]])
Y = asarray([[0., -1.j],[1.j, 0.]])
Z = asarray([[1., 0.],[0.,-1.]])
XX = kron(X,X)
YY = kron(Y,Y)
ZZ = kron(Z,Z)
XI = kron(X,I)
IX = kron(I,X)
ZI = kron(Z,I)
IZ = kron(I,Z)

def tensor_product(operators):
    return reduce(kron, operators)

def get_nlayers(degree, n_repetitions, n_orbitals=None, n_id_layers=0, hamiltonian='fermi-hubbard-1d'):
    if hamiltonian=='fermi-hubbard-1d':
        if degree==1:
            n_SN_layers = 4*n_repetitions
        elif degree==2:
            n_SN_layers = 3*degree*n_repetitions  # Number of layers in pure swap network
            n_SN_layers -= (degree*n_repetitions-1)  # If we absorb 
        elif degree==4:
            n_SN_layers = 20*n_repetitions+1
    elif hamiltonian=='molecular':
        assert(type(n_orbitals) is not type(None))
        if degree in [1,2]:
            n_SN_layers = 2*n_orbitals*n_repetitions  # Number of layers in pure swap network
            n_SN_layers -= (2*n_repetitions-1)  # If we absorb 
        elif degree==4:
            n_SN_layers = 10*(n_orbitals-1)*n_repetitions+1
    elif hamiltonian in ['ising-1d','heisenberg']:
        if degree==1:
            n_SN_layers = 2*n_repetitions
        elif degree==2: 
            n_SN_layers = 2*n_repetitions+1
        elif degree==4:
            n_SN_layers = 10*n_repetitions+1
    return n_SN_layers+n_id_layers

def get_initial_gates(n_sites, t, n_repetitions=1, degree=2, 
                      hamiltonian='fermi-hubbard-1d', n_id_layers=0, use_TN=True, **kwargs):
    if hamiltonian == 'fermi-hubbard-1d':
        T, V = kwargs['T'], kwargs['V']
        Vlist_start = get_swap_network_trotter_gates_fermi_hubbard_1d(T, V, t, n_sites, n_repetitions, degree, use_TN)
        first_layer_odd=False
    elif hamiltonian=='molecular':
        T, V = kwargs['T'], kwargs['V']
        Vlist_start = get_swap_network_trotter_gates_molecular(T, V, t, n_sites, degree, n_repetitions, use_TN=use_TN)
        assert(n_id_layers==0)
    elif hamiltonian in ['ising-1d', 'heisenberg']:
        Vlist_start = get_brickwall_trotter_gates_spin_chain(t, n_sites, n_repetitions, degree, hamiltonian, use_TN, **kwargs)
        if degree==1: first_layer_odd=True
        elif degree in [2,4]: first_layer_odd=False

    if n_id_layers>0:
        Vlist_start = list(Vlist_start)+list(get_identity_layers(n_sites, n_id_layers, first_layer_odd, use_TN))
    return asarray(Vlist_start)

def get_gates_per_layer(Vlist, n_sites, degree=None, n_repetitions=None,
                        n_layers=None, n_id_layers=0, hamiltonian='fermi-hubbard-1d'):
    N_odd_gates, N_even_gates = int(n_sites/2), int(n_sites/2)  # Number of gates per layer
    if n_sites%2==0: N_even_gates -= 1
    if type(n_layers) is type(None):
        assert(type(degree) is not None and type(n_repetitions) is not None)
        n_SN_layers = get_nlayers(degree, n_repetitions, n_sites, n_id_layers, hamiltonian)
    else:
        n_SN_layers = n_layers
        
    if hamiltonian=='fermi-hubbard-1d':
        odd = False  # First layer is even
        lim1, lim2 = 0, N_even_gates
    elif hamiltonian in ['molecular', 'ising-1d', 'heisenberg']:
        odd = True  # First layer is odd
        lim1, lim2 = 0, N_odd_gates

    gates_per_layer, layer_is_odd = [], []
    for _ in range(1, n_SN_layers+1):
        layer_is_odd.append(odd)
        gates_per_layer.append(Vlist[lim1:lim2])
        lim1=lim2; lim2+=N_even_gates if odd else N_odd_gates
        odd = not odd  # Parity of next layer
            
    return gates_per_layer, layer_is_odd