import os
from pickle import load

from numpy.random import uniform, seed
from jax.numpy import array, eye, kron, asarray, ones, zeros, cos, sin, exp, arange, append
from jax import config as c
c.update("jax_enable_x64", True)

from .util import swap

# Define some operators
number_op = array([[0.,0.],[0.,1.]])
a_op = array([[0.,1.],[0.,0.]])
ah_op = array([[0.,0.],[1.,0.]])
Z_op = asarray([[1.,0.],[0.,-1.]])
fermionic_swap = asarray([[1.,0.,0.,0.],[0.,0.,1.,0.],[0.,1.,0.,0.],[0.,0.,0.,-1.]])

def get_nn_operator(u, p, N):
    '''
    $u \ cdot n_{p\arrowup}n_{p\arrowdown}$
    p: spatial orbital starting from 1...N/2
    N: number of spin orbitals
    '''
    i = 2*(p-1)  # Index in chain
    M_bef, M_aft = eye(2**i), eye(2**(N-2*p))
    nn = kron(number_op,number_op)
    M = kron(kron(M_bef, nn), M_aft)
    return u*M

def get_aha_operator(t, i, j, N):
    '''
    $t \cdot (a_i^\dagger a_j + a_j^\dagger a_i)$
    i,j: index of spin orbital in chain
    N: number of spin orbitals
    '''
    A = asarray([[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,0.,0.]])
    B = asarray([[0.,0.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]])

    C = kron(kron(kron(ah_op@Z_op, Z_op), Z_op), a_op)
    D = kron(kron(kron(Z_op@a_op, Z_op), Z_op), ah_op)
    
    M_bef, M_aft = eye(2**i), eye(2**(N-j-1))
    if abs(j-i)==1: op1=A; op2=B
    elif abs(j-i)==3: op1=C; op2=D

    M1 = kron(kron(M_bef, op1), M_aft)
    M2 = kron(kron(M_bef, op2), M_aft)
    
    return -t*(M1+M2)

def construct_spinful_FH1D_hamiltonian(n_orbitals, get_matrix=True, disordered=False, random=False, reference_seed=1235773):
    ''' 
    Construct a spinful disorderd/non-disordered Fermi-Hubbard 1D Hamiltonian.
    Return the Hamiltonian if get_matrix=True and the coefficients for the 
    swap network.
    '''
    seed(reference_seed)
    n_spin_orbitals = 2*n_orbitals
    kin_pairs_nn = array([[i,i+1] for i in range(1,n_spin_orbitals-1,2)])
    kin_pairs_nnnn = array([[i,i+3] for i in range(0,n_spin_orbitals-3,2)])
    assert len(kin_pairs_nn)==len(kin_pairs_nnnn)

    # Get random parameters
    u_coeff, t_coeff, sigma = 4., 1., 0.5
    if disordered:
        t_coeffs = uniform(t_coeff-sigma, t_coeff+sigma, size=(len(kin_pairs_nn)))
        u_coeffs = uniform(u_coeff-sigma, u_coeff+sigma, size=(n_orbitals,))
    else:
        if random:
            t_coeffs = uniform(t_coeff-sigma, t_coeff+sigma)*ones((len(kin_pairs_nn),))
            u_coeffs = uniform(u_coeff-sigma, u_coeff+sigma)*ones((n_orbitals,))
        else:
            t_coeffs = t_coeff*ones((len(kin_pairs_nn),))
            u_coeffs = u_coeff*ones((n_orbitals,))
        
    if get_matrix: H = zeros((2**n_spin_orbitals, 2**n_spin_orbitals))
    else: H = None

    V = zeros((n_orbitals,))
    T = zeros((n_orbitals,n_orbitals))

    for p in range(1,n_orbitals+1):  # p denotes spatial orbital
        u = u_coeffs[p-1]
        V = V.at[p-1].set(u)
        if get_matrix: H += get_nn_operator(u, p, n_spin_orbitals)
    for p,pairs in enumerate(kin_pairs_nn):
        i,j = pairs  # i,j denote spin orbitals
        t = t_coeffs[p]
        T = T.at[p,p+1].set(t)
        T = T.at[p+1,p].set(t)
        if get_matrix: H += get_aha_operator(t, i, j, n_spin_orbitals)
    for p,pairs in enumerate(kin_pairs_nnnn):
        i,j = pairs  # i,j denote spin orbitals
        t = t_coeffs[p]
        if get_matrix: H += get_aha_operator(t, i, j, n_spin_orbitals)

    return H, T, V
        
def kinetic_simulation_gate(Tpq, t): 
    FSG = array([
        [1., 0., 0., 0.],
        [0., cos(-Tpq*t), -1j*sin(-Tpq*t), 0.],
        [0., -1j*sin(-Tpq*t), cos(-Tpq*t), 0.],
        [0., 0., 0., 1]
        ])
    return FSG

def interaction_simulation_gate(Vpq, t, include_swap=True): 
    if include_swap:
        FSG = array([
            [1., 0., 0., 0.],
            [0., 0., 1., 0.],
            [0., 1., 0., 0.],
            [0., 0., 0., -exp(-1j*Vpq*t)]
            ])
    else:
        FSG = array([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., exp(-1j*Vpq*t)]
            ])
    return FSG

def get_swap_network_trotter_gates_fermi_hubbard_1d(
        T, V, t, n_sites, n_repetitions=1, degree=2, use_TN=False):
    
    dt = t/n_repetitions
    if degree in [1,2]:
        dt = dt/degree
        # Get the indices of odd/ even pairs
        odd_pairs = asarray([(i,i+1) for i in range(0,n_sites-1,2)])
        even_pairs = asarray([(i,i+1) for i in range(1,n_sites-1,2)])
        chain = arange(n_sites)

        # Kinetic coefficients
        even_orbital_pairs = [asarray(chain[p]) for p in even_pairs]
        T_coeff = asarray([T[pair[0]//2, pair[1]//2] for pair in even_orbital_pairs])
        T_gate = [kinetic_simulation_gate(Tpq, dt) for Tpq in T_coeff]
        T_gate_squared = [fsg@fsg for fsg in T_gate]
        # Interaction coefficients
        V_gate = [interaction_simulation_gate(Vpq, dt, include_swap=True) for Vpq in V]

        if degree==1:
            gates = T_gate.copy() + V_gate.copy() + T_gate.copy() + [fermionic_swap for _ in V_gate]
            for _ in range(n_repetitions-1):
                gates += gates.copy()
        elif degree==2:
            gates = T_gate.copy() + V_gate.copy() + T_gate_squared.copy() + V_gate.copy()
            for _ in range(n_repetitions-1):
                gates += T_gate_squared.copy() + V_gate.copy() + T_gate_squared.copy() + V_gate.copy()
            gates += T_gate.copy()
        gates = asarray(gates)
        if use_TN: gates = gates.reshape((len(gates),2,2,2,2))  # Flatten the array
        return gates

    elif degree==4:
        lim = int(n_sites/2)-1  # Number of gates in laster/first layer (of order 2), i.e., even layer
        s2 = (4-4**(1/3))**(-1)

        # V1 = U_2(s_2*t)
        V1 = get_swap_network_trotter_gates_fermi_hubbard_1d(T, V, s2*dt, n_sites, n_repetitions=1, degree=2)
        # V2 = U_2((1-4*s_2)*t)
        V2 = get_swap_network_trotter_gates_fermi_hubbard_1d(T, V, (1-4*s2)*dt, n_sites, n_repetitions=1, degree=2)
        
        # Merge the last and first layers of V1, V2
        V1_ = asarray([V1[j]@V1[j] for j in range(lim)])
        V2_ = asarray([V1[j]@V2[j] for j in range(lim)])
        V3_ = asarray([V2[j]@V1[j] for j in range(lim)])

        # First get the gates for one repetition    
        gates_ = append(V1[:-lim], V1_, axis=0)  # Last gate layer is applied twice
        gates_ = append(gates_, V1[lim:], axis=0)  # Leave out first gate layer
        V1_list = gates_.copy()  # U_2(s_2*t) with reduced layers
        gates_ = append(gates_[:-lim], V2_, axis=0)
        gates_ = append(gates_, V2[lim:], axis=0)  # Leave out first gate layer
        gates_ = append(gates_[:-lim], V3_, axis=0)  # Last gate layer is applied twice
        gates_ = append(gates_, V1_list[lim:], axis=0)  # Leave out first gate layer

        # Now get the gates for n_repetition>1
        if n_repetitions==1: 
            gates = asarray(gates_)
        else:
            gates = append(gates_[:-lim], V1_, axis=0)  # Layer 1
            if n_repetitions>2: 
                gates_rep = append(gates_[lim:-lim], V1_, axis=0)  # Part to repeat
                for _ in range(n_repetitions-2):
                    gates = append(gates, gates_rep, axis=0)
            gates = append(gates, gates_[lim:], axis=0)  # Last layer

        if use_TN: gates = gates.reshape((len(gates),2,2,2,2))
        return gates


def load_molecular_model(fdir, load_which, molecule):
    fname =  str(load_which)+'_'+molecule+'.pkl'
    with open(os.path.join(fdir,fname), 'rb') as fp:
        model = load(fp)
        n_orbitals = model['n_orbitals']
        t = model['t']
        H_matrix = model['H_matrix']
        T = model['T']
        V = model['V']
        U_ref = model['U_ref']
        U_mpo = model['U_mpo']
        SN = model['SN']
        n_rotations = model['n_rotations']
        return n_orbitals, t, H_matrix, T, V, U_ref, U_mpo, SN, n_rotations

def get_swap_network_interactions(T, V, n_orbitals, degree=2):
    """
    Returns the list of two-orbital interaction pairs per layer.
    Returns the list of two-orbital interaction coefficients per layer

    """
    # Get the indices of odd/ even pairs
    odd_pairs = asarray([(i,i+1) for i in range(0,n_orbitals-1,2)])
    even_pairs = asarray([(i,i+1) for i in range(1,n_orbitals-1,2)])
        
    chain = arange(n_orbitals)
    qubit_pair_history, T_coeff_history, V_coeff_history = [], [], []
    gate_index_history = []
    lim0=0
    for deg in range(degree):
        for i in range(1, n_orbitals+1):
            pairs=even_pairs if i%2==0 else odd_pairs
            if n_orbitals%2==0 and deg>0: pairs=even_pairs if i%2==1 else odd_pairs
            qubit_pairs = [asarray(chain[p]) for p in pairs]
            T_coeff_history.append(
                asarray([T[pair[0], pair[1]] for pair in qubit_pairs]))
            V_coeff_history.append(
                asarray([V[pair[0], pair[1]] for pair in qubit_pairs]))
            qubit_pair_history.append(qubit_pairs)
    
            # Now actually swap the orbitals in chain
            for p in pairs: chain = swap(chain, p[0], p[1])        

    return qubit_pair_history, T_coeff_history, V_coeff_history, gate_index_history

def reverse_fermionic_swaps_matrix(n_orbitals):  
    # This function reverses all fermionic swaps for a given number of qubits
    
    swap_total = eye(2**n_orbitals)
    for layer in range(n_orbitals):     
        odd=True if layer%2==0 else False  # Odd case
        if odd:
            swap_ = fermionic_swap.copy()
            N = int((n_orbitals-1)/2) if n_orbitals%2==1 else int(n_orbitals/2)
            N = N-1
        else:
            swap_ = eye(2)
            N = int((n_orbitals-2)/2) if (n_orbitals-1)%2==1 else int((n_orbitals-1)/2)
            
        for _ in range(N):
            swap_ = kron(swap_, fermionic_swap)
        if n_orbitals%2==int(odd): swap_ = kron(swap_, eye(2))
    
        swap_total = swap_ @ swap_total
    
    return swap_total

def fermionic_simulation_gate(Tpq, Vpq, t, include_swap=True): 
    if include_swap:
        FSG = array([
            [1., 0., 0., 0.],
            [0., -1j*sin(Tpq*t), cos(Tpq*t), 0.],
            [0., cos(Tpq*t), -1j*sin(Tpq*t), 0.],
            [0., 0., 0., -exp(-1j*Vpq*t)]
            ])
    else:
        FSG = array([
            [1., 0., 0., 0.],
            [0., cos(Tpq*t), -1j*sin(Tpq*t), 0.],
            [0., -1j*sin(Tpq*t), cos(Tpq*t), 0.],
            [0., 0., 0., exp(-1j*Vpq*t)]
            ])
    return FSG

def get_swap_network_trotter_gates_molecular(T, V, t, n_orbitals, degree=2, n_repetitions=1, 
                                             return_layers=False, use_TN=False):
    '''
    Get an array of two-qubit gates for the swap network for a Trotterization of order I, II, and IV.
    '''
    dt = t/n_repetitions 

    if degree in [1,2]:
        dt = dt/degree
        _, T_coeffs_history, V_coeffs_history, _ = get_swap_network_interactions(T, V, n_orbitals, degree=1)

        T_coeffs_start, V_coeffs_start = T_coeffs_history[0], V_coeffs_history[0]
        T_coeffs_end, V_coeffs_end = T_coeffs_history[-1], V_coeffs_history[-1]

        first_layer_with_swap = [fermionic_simulation_gate(
            Tpq, Vpq, dt, include_swap=True) for Tpq, Vpq in zip(T_coeffs_start,V_coeffs_start)]
        first_layer_no_swap = [fermionic_simulation_gate(
            Tpq, Vpq, dt, include_swap=False) for Tpq, Vpq in zip(T_coeffs_start,V_coeffs_start)]
        last_layer_no_swap = [fermionic_simulation_gate(
            Tpq, Vpq, dt, include_swap=False) for Tpq, Vpq in zip(T_coeffs_end,V_coeffs_end)]
        
        layers, layers_inter = [first_layer_with_swap.copy()], []

        # Get gates for simple swap network without repetition and without last layer
        for T_coeffs, V_coeffs in zip(T_coeffs_history[1:-1], V_coeffs_history[1:-1]):
            layer = [fermionic_simulation_gate(Tpq, Vpq, dt, include_swap=True) for Tpq, Vpq in zip(T_coeffs, V_coeffs)]
            layers_inter.append(layer)
        
        if degree==1:
            layers_inter.append(last_layer_no_swap)  # Last layer
            reverse_swap_layers = []
            for layer in reversed(layers_inter[:-1]):
                reverse_swap_layer = [fermionic_swap for _ in range(len(layer))]
                reverse_swap_layers.append(reverse_swap_layer)
            layers_inter = layers_inter + reverse_swap_layers

            layers += layers_inter  # First repetition
            for _ in range(n_repetitions-1):
                layers += [first_layer_no_swap]
                layers += layers_inter
            layers.append([fermionic_swap for _ in range(len(layers[0]))])
    
        elif degree==2:
            first_layer_no_swap_2t = [fermionic_simulation_gate(
                Tpq, Vpq, 2*dt, include_swap=False) for Tpq, Vpq in zip(T_coeffs_start,V_coeffs_start)]
            last_layer_no_swap_2t = [fermionic_simulation_gate(
                Tpq, Vpq, 2*dt, include_swap=False) for Tpq, Vpq in zip(T_coeffs_end,V_coeffs_end)]
            
            reverse_layers = [last_layer_no_swap_2t]  # Reversed direction for second order
            for T_coeffs, V_coeffs in zip(reversed(T_coeffs_history[1:-1]), reversed(V_coeffs_history[1:-1])):
                reverse_layer = [fermionic_simulation_gate(Tpq, Vpq, dt, include_swap=True) for Tpq, Vpq in zip(T_coeffs, V_coeffs)]
                reverse_layers.append(reverse_layer)
            layers_inter = layers_inter + reverse_layers

            layers += layers_inter  # First repetition
            for _ in range(n_repetitions-1):
                layers += [first_layer_no_swap_2t]
                layers += layers_inter
            layers += [first_layer_with_swap]

        if return_layers: return layers
    
    elif degree==4:
        s2 = (4-4**(1/3))**(-1)

        # U_2(s_2*t)^2
        V1_layers = get_swap_network_trotter_gates_molecular(
            T, V, 2*s2*dt, n_orbitals, degree=2, n_repetitions=2, return_layers=True, use_TN=False)
        # U_2((1-4*s_2)*t)
        V2_layers = get_swap_network_trotter_gates_molecular(
            T, V, (1-4*s2)*dt, n_orbitals, degree=2, n_repetitions=1, return_layers=True, use_TN=False)

        # Merged layers
        V1_V2_layer = [g1@g2 for g1,g2 in zip(V1_layers[-1], V2_layers[0])]
        V2_V1_layer = [g1@g2 for g1,g2 in zip(V2_layers[-1], V1_layers[0])]
        V1_V1_layer = [g1@g2 for g1,g2 in zip(V1_layers[-1], V1_layers[0])]
        
        # Repeated layers
        layers_inter = V1_layers[1:-1] + [V1_V2_layer] + V2_layers[1:-1] + [V2_V1_layer] + V1_layers[1:-1]

        # Combine all layers
        layers = [V1_layers[0]] + layers_inter
        for _ in range(n_repetitions-1):
            layers += [V1_V1_layer]
            layers += layers_inter
        layers += [V1_layers[-1]]
        
    else: raise Exception(f'Trotter order {degree} not implemented.')
    gates = asarray([g for G in layers for g in G])
    if use_TN: gates = gates.reshape((len(gates),2,2,2,2))
    return gates