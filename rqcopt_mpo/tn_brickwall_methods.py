import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

from .brickwall_circuit import get_gates_per_layer
from .tn_helpers import (get_id_mpo, canonicalize_local_tensor, merge_two_mpos_and_gate, 
                         split_tensor_into_half_canonical_mpo_pair, right_to_left_RQ_sweep,
                         merge_mpo_and_layer, compress_mpo, fully_contract_mpo)
from .util import project_unitary_tangent_vectorized

def get_mpo_pairs(mpo, odd):
    """
    Get the MPO pairs for merging with a layer if a layer is 
    odd=True or even (odd=False).
    """
    if not odd: mpo_pairs = [(mpo[0],)]; start=1
    else: mpo_pairs = []; start=0
    for impo in range(start, len(mpo)-1, 2):
        mpo_pairs.append((mpo[impo], mpo[impo+1]))
    if not odd and len(mpo)%2==0: mpo_pairs.append((mpo[-1],))
    elif odd and len(mpo)%2==1: mpo_pairs.append((mpo[-1],))
    return mpo_pairs

def merge_and_truncate_mpo_and_layer_right_to_left(mpo_init, gates_in_layer, odd_layer=True, 
                                                   max_bondim=128, layer_is_below=True):
    '''
    layer_is_below corresponds to layer_is_left
    '''
    n_spin_orbitals=len(mpo_init)
    if not odd_layer:  # For even layers
        Q, R = canonicalize_local_tensor(mpo_init[-1], left=False)
        mpo_res = [Q]; i_mpo_init=n_spin_orbitals-2
    else:
        mpo_res=[]; i_mpo_init=n_spin_orbitals-1

    while i_mpo_init-1>=0:
        mpo1, mpo2 = mpo_init[i_mpo_init-1], mpo_init[i_mpo_init]
        if i_mpo_init==n_spin_orbitals-1: mpo2_R = mpo2
        else: mpo2_R = jnp.einsum('iabj,jk->iabk', mpo2, R)  # Merge the R tensor in last local mpo
        gate = gates_in_layer[int((i_mpo_init-1)/2)]
        merged_T = merge_two_mpos_and_gate(gate, mpo1, mpo2_R, gate_is_left=layer_is_below)  # Merge gate and local tensor pair
        T1, T2 = split_tensor_into_half_canonical_mpo_pair(merged_T, canonical_mode='right', max_bondim=max_bondim)
        if i_mpo_init-1==0:
            mpo_res += [T2, T1]  # T2, T1
        else: 
            Q, R = canonicalize_local_tensor(T1, left=False)
            mpo_res += [T2, Q]
        i_mpo_init -= 2
    
    if i_mpo_init==0: 
            mpo_R = jnp.einsum('iabj,jk->iabk', mpo_init[0], R)
            mpo_res += [mpo_R] 

    mpo_res = mpo_res[::-1]  # Reverse the order of MPO
    return mpo_res

def merge_and_truncate_mpo_and_layer_left_to_right(mpo_init, gates_in_layer, odd_layer=False, 
                                                   max_bondim=128, layer_is_below=True):
    """
    Function to merge and compress a layer of two-qubit gates with an MPO.
    The sweep is done from left to right using RQ decomposition.
    """
    n_spin_orbitals=len(mpo_init)
    if not odd_layer:  # For even layers
        Q, R = canonicalize_local_tensor(mpo_init[0], left=True)
        mpo_res = [Q]; i_mpo_init=1
    else:
        mpo_res=[]; i_mpo_init=0

    while i_mpo_init+1<n_spin_orbitals:
        mpo1, mpo2 = mpo_init[i_mpo_init:i_mpo_init+2]
        if i_mpo_init==0: mpo1_R = mpo1
        else: mpo1_R = jnp.einsum('ij,jabk->iabk', R, mpo1)  # Merge the R tensor in first local mpo
        gate = gates_in_layer[int(i_mpo_init/2)]
        merged_T = merge_two_mpos_and_gate(gate, mpo1_R, mpo2, gate_is_left=layer_is_below)  # Merge gate and local tensor pair
        T1, T2 = split_tensor_into_half_canonical_mpo_pair(merged_T, canonical_mode='left', max_bondim=max_bondim)
        if i_mpo_init+1==n_spin_orbitals-1:
            mpo_res += [T1, T2]
        else: 
            Q, R = canonicalize_local_tensor(T2, left=True)
            mpo_res += [T1, Q]
        i_mpo_init += 2

    if i_mpo_init==n_spin_orbitals-1: 
        mpo_R = jnp.einsum('ij,jabk->iabk', R, mpo_init[-1])
        mpo_res += [mpo_R]        

    return mpo_res

def contract_layers_of_swap_network_with_mpo(mpo_init, gates_per_layer, layer_is_odd, 
                                             layer_is_left=True, max_bondim=128,
                                             get_norm=False):
    '''
    Method to merge and compress the swap network with the reference MPO
    layerwise.
    '''
    nlayers = len(gates_per_layer)
    mpo = mpo_init.copy()
    if layer_is_left: iterator=reversed(range(nlayers)) 
    else: iterator=range(nlayers)

    # Bring initial MPO into right canonical form
    mpo = right_to_left_RQ_sweep(mpo, get_norm=False)
    merge_left_to_right = True

    for layer in iterator:
        # Get the gates in the layer
        odd = layer_is_odd[layer]
        gates = gates_per_layer[layer]
        # Merge layer with MPO
        if merge_left_to_right: mpo = merge_and_truncate_mpo_and_layer_left_to_right(mpo, gates, odd, max_bondim, layer_is_left)
        else: mpo = merge_and_truncate_mpo_and_layer_right_to_left(mpo, gates, odd, max_bondim, layer_is_left)
        merge_left_to_right = not merge_left_to_right

    # Bring MPO into right canonical form and obtain its norm (unitarity check)
    if get_norm:
        mpo, nrm = right_to_left_RQ_sweep(mpo, get_norm=True)
        return mpo, nrm

    else: 
        return mpo

def contract_layers_of_swap_network(
        mpo_init, gates_per_layer, layer_is_odd, 
        layer_is_left=True, max_bondim=128):
    '''
    Yu's approach: method='QR-SVD'
    Gray's approach: method='QR-RQ-SVD'

    gates_per_layer: list of list per layer with Vlist_TN per layer

    This function takes an initial MPO and contracts it with
    the list of layers given. This is used to obtain the fragmented reference.

    The gates are splitted into 1-qubit tensors in this function.
    '''
    
    nlayers = len(gates_per_layer)
    iterator=reversed(range(nlayers)) if layer_is_left else range(nlayers)

    for layer in iterator:      
        odd = layer_is_odd[layer]
        gates = gates_per_layer[layer]

        # Merge layer with MPO
        mpo_res = merge_mpo_and_layer(gates, mpo_init, odd, layer_is_left=True)
        
        # Truncate the resulting MPO
        mpo_res = compress_mpo(mpo_res, max_bondim)
        mpo_init = mpo_res.copy()

    return mpo_res

def fully_contract_swap_network_mpo(
        Vlist_TN, U_mpo, degree, n_repetitions, n_id_layers, n_layers,
        max_bondim, hamiltonian):
    
    n_sites = len(U_mpo)
    
    gates_per_layer, layer_is_odd = get_gates_per_layer(
        Vlist_TN, n_sites, degree, n_repetitions, n_layers, n_id_layers, hamiltonian)
    
    layer_is_left = True
    mpo_res = contract_layers_of_swap_network_with_mpo(U_mpo, gates_per_layer, layer_is_odd,
                                                       layer_is_left, max_bondim)
    trace = fully_contract_mpo(mpo_res)
    return trace
    
def compute_partial_derivatives_in_layer(gates_in_layer, layer_odd, upper_env_mpo, lower_env_mpo):
    if layer_odd:  # If the layer is odd, the edge qubits have a gate acting on it
        i_mpo = len(upper_env_mpo)
        R, L = jnp.eye(1), jnp.eye(1)
    else:  # Otherwise, the edge environments are given by the local MPOs
        i_mpo = len(upper_env_mpo)-1
        R = jnp.einsum('abcd,ecbd->ae', upper_env_mpo[-1], lower_env_mpo[-1])
        L = jnp.einsum('abcd,acbe->de', upper_env_mpo[0], lower_env_mpo[0])
    R_envs = [R.copy()]

    # Compute all right environments (go from right to left)
    for gate in reversed(gates_in_layer[1:]):
        A1, A2, B1, B2 = *upper_env_mpo[i_mpo-2:i_mpo], *lower_env_mpo[i_mpo-2:i_mpo]
        R = jnp.einsum('abcd,defg,cfhk,ihbj,jkel,gl->ai', A1, A2, gate, B1, B2, R)
        R_envs.append(R.copy())
        i_mpo -= 2
    R_envs = R_envs[::-1]

    # Compute now all partial derivatives for an odd layer (go from left to right)
    grad = jnp.empty_like(gates_in_layer)
    if layer_odd: i_mpo, i_env_mpo = 0, 0
    else: i_mpo, i_env_mpo = 1, 1
    for cut_out_gate in range(len(gates_in_layer)):
        # Get local MPO tensors of lower and upper environments
        A1, A2, B1, B2 = *upper_env_mpo[i_mpo:i_mpo+2], *lower_env_mpo[i_mpo:i_mpo+2]
        i_mpo += 2

        # Compute R, L
        if cut_out_gate>0:  # R,L for the very left gate are already computed
            # Current R
            R = R_envs[cut_out_gate]  
            # Current L
            L = jnp.einsum('ai,abcd,defg,cfhk,ihbj,jkel->gl', L, *upper_env_mpo[i_env_mpo:i_env_mpo+2], 
                           gates_in_layer[cut_out_gate-1], *lower_env_mpo[i_env_mpo:i_env_mpo+2])
            i_env_mpo += 2

        # Contract everything
        res = jnp.einsum('ab,acde,efgh,bick,kjfl,hl->dgij', L, A1, A2, B1, B2, R)
        grad = grad.at[cut_out_gate].set(res)
    return grad.conj()

def compute_full_gradient(U_mpo, gates_per_layer, layer_is_odd, max_bondim, compute_overlap=False):
    # Initialize edge environments
    bottom_env, top_env = get_id_mpo(len(U_mpo)), U_mpo.copy()
    upper_envs, grad = [top_env.copy()], []
    merge_right_to_left = True  # Assume a left canonicalized reference

    # Compute all upper environments (go from top to bottom)
    for gates, odd in zip(reversed(gates_per_layer[1:]), reversed(layer_is_odd[1:])):
        # Merge all layers up to the last one (from left to right) of the swap network into the initial id MPO
        if merge_right_to_left: top_env = merge_and_truncate_mpo_and_layer_right_to_left(
            top_env, gates, odd, max_bondim, layer_is_below=True)
        else: top_env = merge_and_truncate_mpo_and_layer_left_to_right(
            top_env, gates, odd, max_bondim, layer_is_below=True)
        upper_envs.append(top_env.copy())
        merge_right_to_left = not merge_right_to_left
    upper_envs = upper_envs[::-1]

    # Now compute the gradient
    for layer in range(len(gates_per_layer)):
        if layer>0: 
            #odd = layer_is_odd[layer-1]
            if merge_right_to_left: bottom_env = merge_and_truncate_mpo_and_layer_right_to_left(
                bottom_env, gates_per_layer[layer-1], layer_is_odd[layer-1], max_bondim, layer_is_below=False)
            else: bottom_env = merge_and_truncate_mpo_and_layer_left_to_right(
                bottom_env, gates_per_layer[layer-1], layer_is_odd[layer-1], max_bondim, layer_is_below=False)
            merge_right_to_left = not merge_right_to_left

        # Compute the partial derivatives in this layer
        grad += list(compute_partial_derivatives_in_layer(gates_per_layer[layer], layer_is_odd[layer], upper_envs[layer], bottom_env))

    grad = jnp.asarray(grad)
    if compute_overlap: 
        overlap = jnp.einsum('abcd,abcd->', grad[0].conj(), gates_per_layer[0][0])  # Compute tr(U^\dagger W)
        return grad, overlap 
    else: return grad

def get_riemannian_gradient_and_cost_function(U_mpo, Vlist_TN, n_sites, degree, n_repetitions, n_id_layers,
                                              max_bondim, reference_is_normalized, hamiltonian):
    #if hamiltonian=='fermi-hubbard-1d': n_sites = 2*n_orbitals
    #elif hamiltonian=='molecular': n_sites = n_orbitals
    gates_per_layer, layer_is_odd = get_gates_per_layer(Vlist_TN, n_sites, degree, n_repetitions, 
                                                        n_id_layers=n_id_layers, hamiltonian=hamiltonian)
    grad, overlap = compute_full_gradient(U_mpo, gates_per_layer, layer_is_odd, max_bondim, compute_overlap=True)

    # Get Riemannian gradient from Euclidean gradient
    vlist_reshaped = Vlist_TN.reshape((Vlist_TN.shape[0],4,4))
    grad_reshaped = grad.reshape(vlist_reshaped.shape)
    projected_grad = - project_unitary_tangent_vectorized(vlist_reshaped, grad_reshaped, True)

    # Get Frobenius norm and Hilbert-Schmidt test from overlap -> normalization holds for normalized reference!
    if reference_is_normalized: const_F=2**int(n_sites/2)
    else: const_F=2**n_sites
    cost_F = 2 - 2*overlap.real/const_F  # Frobenius norm

    return cost_F, projected_grad