from functools import reduce

from numpy.random import uniform, seed
import jax.numpy as jnp
from jax.scipy.linalg import expm

from jax import config as c
c.update("jax_enable_x64", True)

I = jnp.eye(2)
X = jnp.asarray([[0., 1.],[1., 0.]])
Y = jnp.asarray([[0., -1.j],[1.j, 0.]])
Z = jnp.asarray([[1., 0.],[0.,-1.]])
XX, YY, ZZ = jnp.kron(X,X), jnp.kron(Y,Y), jnp.kron(Z,Z)
XI, IX = jnp.kron(X,I), jnp.kron(I,X)
YI, IY = jnp.kron(Y,I), jnp.kron(I,Y)
ZI, IZ = jnp.kron(Z,I), jnp.kron(I,Z)

def tensor_product(operators):
    return reduce(jnp.kron, operators)

def operator_chain(N, ind, string_op, op, single_qubit=True):
    end=N-ind-1 if single_qubit else N-ind-2
    operators = [string_op for _ in range(ind)]
    operators += [op]
    operators += [string_op for _ in range(end)]
    return tensor_product(operators)


def brickwall_gate(t, is_edge=False, is_top=False, hamiltonian='ising-1d', **kwargs):
    J = kwargs.get('J', 0.)
    g = kwargs.get('g', [0.,0.])
    h = kwargs.get('h', [0.,0.])
    if hamiltonian == 'ising-1d':
        g1, g2, h1, h2 = g[0]/2, g[1]/2, h[0]/2, h[1]/2
        if is_edge: 
            if is_top: g1, h1 = g[0], h[0]
            else: g2, h2 = g[1], h[1]
        gate = expm(-1j * t* (J*ZZ + g1*XI + g2*IX + h1*ZI + h2*IZ))
        return gate
    
    elif hamiltonian=='heisenberg':
        h1, h2 = h[0]/2, h[1]/2
        if is_edge: 
            if is_top: h1 = h[0]
            else: h2 = h[1] 
        op1, op2 = [XX,YY,ZZ], [[XI,IX],[YI,IY],[ZI,IZ]]
        exp = jnp.zeros_like(XX)
        for i in range(3):
            exp += (J[i]*op1[i] + h1[i]*op2[i][0] + h2[i]*op2[i][1])
        gate = expm(-1j*t*exp)
        return gate    


def construct_heisenberg_hamiltonian(N, J=[1,1,-.5], h=[.75,0,0], disordered=False, get_matrix=False, reference_seed=123456):
    seed(reference_seed)
    if disordered: 
        J, h = jnp.asarray(J), jnp.asarray(h)
        sigmaJ, sigmah = J/2, h/2
        _Js = jnp.asarray([uniform(J-sigmaJ, J+sigmaJ) for _ in range(N-1)])
        hs = jnp.asarray([uniform(h-sigmah, h+sigmah) for _ in range(N)])
    else:
        _Js, hs = jnp.asarray([J.copy() for _ in range(N)]), jnp.asarray([h.copy() for _ in range(N)])
    hamiltonian = jnp.zeros(2**N) if get_matrix else None
    Js = jnp.zeros((N,N,3))
    sigma_sigma, sigma = [XX, YY, ZZ], [X, Y, Z]
    for j in range(N-1):
        for i in range(3):
            Js = Js.at[j,j+1,i].set(_Js[j,i])
            if get_matrix:
                hamiltonian += operator_chain(N, j, I, _Js[j,i]*sigma_sigma[i], single_qubit=False)
                hamiltonian += operator_chain(N, j, I, hs[j,i]*sigma[i])
    if get_matrix:
        for i in range(3):
            hamiltonian += operator_chain(N, N-1, I, hs[N-1,i]*sigma[i])
    return hamiltonian, Js, hs



def construct_ising_hamiltonian(N, J=1., g=0.75, h=0.6, disordered=False, get_matrix=False, reference_seed=123456):
    seed(reference_seed)
    if disordered: 
        sigmaJ, sigmag, sigmah = J/2, g/2, h/2
        _Js = uniform(J-sigmaJ, J+sigmaJ, size=(N-1))
        gs = uniform(g-sigmag, g+sigmag, size=(N))
        hs = uniform(h-sigmah, h+sigmah, size=(N))
    else:
        _Js, gs, hs = jnp.ones(N-1)*J, jnp.ones(N)*g, jnp.ones(N)*h
    hamiltonian = jnp.zeros(2**N) if get_matrix else None
    Js = jnp.zeros((N,N))
    for j in range(N-1):
        Js = Js.at[j,j+1].set(_Js[j])
        if get_matrix:
            hamiltonian += operator_chain(N, j, I, _Js[j]*ZZ, single_qubit=False)
            hamiltonian += (operator_chain(N, j, I, gs[j]*X) + operator_chain(N, j, I, hs[j]*Z))
    if get_matrix: hamiltonian += (operator_chain(N, N-1, I, gs[N-1]*X) + operator_chain(N, N-1, I, hs[N-1]*Z))
    
    return hamiltonian, Js, gs, hs



def get_brickwall_trotter_gates_spin_chain(t, n_sites, n_repetitions=1, degree=2, hamiltonian='ising-1d', use_TN=False, **kwargs):
    """
    Return the brickwall circuit gates for Ising.
    Only implemented for even number of sites.

    """
    dt = t/n_repetitions
    J = kwargs.get('J', 0.)
    h = kwargs.get('h', 0.)
    g = kwargs.get('g', 0.)

    N_odd_gates, N_even_gates = int(n_sites/2), int(n_sites/2)  # Number of gates per layer
    if n_sites%2==0: N_even_gates -= 1
    odd_pairs = jnp.array([[i,i+1] for i in range(0,n_sites-1,2)])
    even_pairs = jnp.array([[i,i+1] for i in range(1,n_sites-1,2)])

    if degree in [1,2]:
        dt = dt/degree
        if hamiltonian=='ising-1d':
            gate_1 = brickwall_gate(dt, is_edge=True, is_top=True, hamiltonian=hamiltonian,
                                    J=J[tuple(odd_pairs[0])], g=g[odd_pairs[0]], h=h[odd_pairs[0]])  # First edge gate
            gate_3 = brickwall_gate(dt, is_edge=True, is_top=False, hamiltonian=hamiltonian, 
                                    J=J[tuple(odd_pairs[-1])], g=g[odd_pairs[-1]], h=h[odd_pairs[-1]])  # Last edge gate
            middle_gate = lambda _J, _g, _h: brickwall_gate(
                dt, is_edge=False, is_top=False, hamiltonian=hamiltonian, J=_J, g=_g, h=_h) 
        elif hamiltonian=='heisenberg':
            gate_1 = brickwall_gate(dt, is_edge=True, is_top=True, hamiltonian=hamiltonian, 
                                    J=J[tuple(odd_pairs[0])], h=h[odd_pairs[0]])  # First edge gate
            gate_3 = brickwall_gate(dt, is_edge=True, is_top=False, hamiltonian=hamiltonian, 
                                    J=J[tuple(odd_pairs[-1])], h=h[odd_pairs[-1]])  # Last edge gate
            middle_gate = lambda _J, _h: brickwall_gate(
                dt, is_edge=False, is_top=False, hamiltonian=hamiltonian, J=_J, h=_h) 

        # First layer is odd
        Js_middle_odd = jnp.asarray([J[tuple(pair)] for pair in odd_pairs[1:-1]])
        Js_middle_even = jnp.asarray([J[tuple(pair)] for pair in even_pairs])
        hs_middle_odd = jnp.asarray([h[pair] for pair in odd_pairs[1:-1]])
        hs_middle_even = jnp.asarray([h[pair] for pair in even_pairs])
        if hamiltonian=='ising-1d':
            gs_middle_odd = jnp.asarray([g[pair] for pair in odd_pairs[1:-1]])
            gs_middle_even = jnp.asarray([g[pair] for pair in even_pairs])
            middle_gates_odd = [middle_gate(_J,_g,_h) for _J,_g,_h in zip(Js_middle_odd,gs_middle_odd,hs_middle_odd)] 
            # Second layer is even and has no edge gates
            L2 = [middle_gate(_J,_g,_h) for _J,_g,_h in zip(Js_middle_even,gs_middle_even,hs_middle_even)]
        elif hamiltonian=='heisenberg':
            middle_gates_odd = [middle_gate(_J,_h) for _J,_h in zip(Js_middle_odd,hs_middle_odd)] 
            # Second layer is even and has no edge gates
            L2 = [middle_gate(_J,_h) for _J,_h in zip(Js_middle_even,hs_middle_even)]
        else: raise Exception('Hamiltonian not implemented')
        L1 = [gate_1.copy()] + middle_gates_odd + [gate_3.copy()]  # First layer

        L1_squared = [(gate_1@gate_1).copy()] + [(gate_2@gate_2).copy() for gate_2 in middle_gates_odd] + [(gate_3@gate_3).copy()]
        L2_squared = [(gate_2@gate_2).copy() for gate_2 in L2]

        if degree==1:
            gates = L1 + L2
            for _ in range(n_repetitions-1):
                gates += gates.copy()
        else:
            gates = L1.copy() + L2_squared.copy()
            for _ in range(n_repetitions-1):
                gates += (L1_squared.copy() + L2_squared.copy())
            gates += L1.copy()

        gates = jnp.asarray(gates)
        if use_TN: gates = gates.reshape((len(gates),2,2,2,2))  # Flatten the array
        return gates

    elif degree==4:
        lim = N_odd_gates  # Number of gates in laster/first layer (of order 2), i.e., odd layer
        s2 = (4-4**(1/3))**(-1)

        # V1 = U_2(s_2*t)
        V1 = list(get_brickwall_trotter_gates_spin_chain(2*s2*dt, n_sites, n_repetitions=2, degree=2, hamiltonian=hamiltonian, **kwargs))
        # V2 = U_2((1-4*s_2)*t)
        V2 = list(get_brickwall_trotter_gates_spin_chain((1-4*s2)*dt, n_sites, n_repetitions=1, degree=2, hamiltonian=hamiltonian, **kwargs))
        
        # Merge the last and first layers of V1, V2
        V11 = [V1[j]@V1[j] for j in range(lim)]
        V12 = [V1[j]@V2[j] for j in range(lim)]
        V21 = [V2[j]@V1[j] for j in range(lim)]

        repeated_gates = V1[lim:-lim] + V12 + V2[lim:-lim] + V21 + V1[lim:-lim]

        gates = V1[:lim].copy() + repeated_gates.copy()
        for _ in range(n_repetitions-1):
            gates += V11.copy()
            gates += repeated_gates.copy()
        gates += V1[:lim]
        gates = jnp.asarray(gates)

        if use_TN: gates = gates.reshape((len(gates),2,2,2,2))
        return gates

