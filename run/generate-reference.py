import sys
from time import time
import os
import yaml

from time import time

from jax.numpy import asarray
from jax.scipy.linalg import expm
from jax import config
config.update("jax_enable_x64", True)

from rqcopt_mpo.save_model import save_reference
from rqcopt_mpo.spin_systems import construct_ising_hamiltonian, construct_heisenberg_hamiltonian
from rqcopt_mpo.fermionic_systems import construct_spinful_FH1D_hamiltonian
from rqcopt_mpo.tn_helpers import (get_id_mpo, convert_mpo_to_mps, get_maximum_bond_dimension, 
                                   get_left_canonical_mps, inner_product_mps, compress_mpo,
                                   get_mpo_from_matrix, get_maximum_bond_dimension)
from rqcopt_mpo.tn_brickwall_methods import contract_layers_of_swap_network_with_mpo
from rqcopt_mpo.brickwall_circuit import get_gates_per_layer, get_initial_gates

from helpers import get_duration

def compute_error_mpo(mpo1, mpo2):
    '''
    Compute the Frobenius norm and Hilbert-Schmidt test.
    '''
    # Convert MPOs to MPSs
    mps1, mps2 = convert_mpo_to_mps(mpo1), convert_mpo_to_mps(mpo2)
    # Normalize the MPS
    mps1_nrmd = get_left_canonical_mps(mps1, normalize=True, get_norm=False)
    mps2_nrmd = get_left_canonical_mps(mps2, normalize=True, get_norm=False)
    # Compute overlap
    tr = inner_product_mps(mps1_nrmd, mps2_nrmd)
    err = 2 - 2*tr.real  # Frobenius norm
    return err

def compute_maximum_duration(config, **kwargs):
    t = config['t_start']
    bond_dim = config['max_bond_dim']
    mpo_id = get_id_mpo(config['n_sites'])
    err = 0.
    while err<=1e-10:
        print('Current time: ', t)
        gates = get_initial_gates(config['n_sites'], t, config['n_repetitions'], config['degree'], 
                                  hamiltonian=config['hamiltonian'], use_TN=True, **kwargs)
        gates_per_layer, layer_is_odd = get_gates_per_layer(
            gates, config['n_sites'], config["degree"], config["n_repetitions"], hamiltonian=config['hamiltonian'])
        mpo_init = contract_layers_of_swap_network_with_mpo(mpo_id, gates_per_layer, layer_is_odd, layer_is_left=True, 
                                                            max_bondim=bond_dim, get_norm=False)
        mpo_reduced = contract_layers_of_swap_network_with_mpo(mpo_id, gates_per_layer, layer_is_odd, layer_is_left=True, 
                                                               max_bondim=bond_dim-1, get_norm=False)
        err = compute_error_mpo(mpo_reduced, mpo_init)
        t += 0.1
    t_end = t - 0.1
    print('Maximum simulation time: ', t_end)
    
def compute_reference(config, ref_batch, ref_seed, **kwargs):
    tstart = time()

    use_full_rank_matrix = config.get('use_full_rank_matrix', False)
    mpo_id = get_id_mpo(config['n_sites'])

    if config['use_full_rank_matrix']:
        H_mat, _, _, _ = construct_ising_hamiltonian(config['n_sites'], config['J'], config['g'], config['h'], 
                                                        disordered=config['disordered'], get_matrix=True)
        U_ref = expm(1j*config['t']*H_mat)  # Adjoint of the time evolution operator
        mpo_init = get_mpo_from_matrix(U_ref)
        bond_dim = get_maximum_bond_dimension(mpo_init)
        print("Maximum bond dimension from full-rank matrix: ", bond_dim)

    else:
        # We start with an initial MPO with maximum bond dimension
        gates = get_initial_gates(config['n_sites'], config['t'], config['n_repetitions'], config['degree'], 
                                hamiltonian=config['hamiltonian'], use_TN=True, **kwargs)
        gates_per_layer, layer_is_odd = get_gates_per_layer(
            gates, config['n_sites'], config["degree"], config["n_repetitions"], hamiltonian=config['hamiltonian'])
        bond_dim = config['max_bond_dim']
        # Obtain MPO representation
        mpo_init = contract_layers_of_swap_network_with_mpo(
            mpo_id, gates_per_layer, layer_is_odd, layer_is_left=True, max_bondim=bond_dim, get_norm=False)

    compress = config.get('compress', True)
    if compress:
        err_threshold = config.get('err_thres', None)
        if type(err_threshold) is type(None):
            degree_thres = config.get('degree_thres', 2)
            n_rep_thres = config.get('n_rep_thres', 10)
            gates_thres = get_initial_gates(
                config['n_sites'], config['t'], n_rep_thres, degree=degree_thres, hamiltonian=config['hamiltonian'], use_TN=True, **kwargs)
            gates_per_layer_thres, layer_is_odd_thres = get_gates_per_layer(
                gates_thres, config['n_sites'], degree=degree_thres, n_repetitions=n_rep_thres, hamiltonian=config['hamiltonian'])
            # Obtain MPO representation
            mpo_thres = contract_layers_of_swap_network_with_mpo(
                mpo_id, gates_per_layer_thres, layer_is_odd_thres, layer_is_left=True, max_bondim=bond_dim, get_norm=False)
            err_threshold = compute_error_mpo(mpo_thres, mpo_init)
            fac_thres = config.get('fac_thres', 500)
            err_threshold = err_threshold/fac_thres
        else:
            err_threshold = float(err_threshold)
        print("\t Error threshold = ", err_threshold)

        # Compress the initial MPO down to convergence criteria
        step_size = 1
        bond_dim_comp = bond_dim
        err2 = 0.
        while err2<err_threshold:
            bond_dim_comp -= step_size
            mpo = compress_mpo(mpo_init, bond_dim_comp)
            err2 = compute_error_mpo(mpo_init, mpo)
            print(f"\t Errors for bond dims {bond_dim_comp}: {err2}")

        # Get information about final reference
        print('Final MPO has maximum bond dimension: ', bond_dim_comp)
        print('\nFinal error between reference MPO and converged MPO: ', err2)
    else:
        print('Final MPO has maximum bond dimension: ', get_maximum_bond_dimension(mpo_init))
        mpo = mpo_init
        err_threshold = None
    
    # Save the reference
    path = os.path.join(os.getcwd(), config['hamiltonian'], "reference")
    _ = save_reference(path, mpo, config["t"], config["n_sites"], config["degree"], config["n_repetitions"], 
                       err_threshold=err_threshold, hamiltonian=config['hamiltonian'], H=None, ref_seed=ref_seed, ref_nbr=ref_batch, **kwargs)
    
    get_duration(tstart, program='cycle')
    print('\n\n')

def main():
    t0 = time()
    # Read in config
    with open(sys.argv[1], 'r') as f:
        config = yaml.safe_load(f)

    # Set modus
    if 'compute_maximum_time' not in config.keys(): config['compute_maximum_time']=False
    if 'compute_reference' not in config.keys(): config['compute_reference']=False
    if config['hamiltonian']=='fermi-hubbard-1d' and 'n_sites' not in config.keys(): config['n_sites']=2*config['n_orbitals']

    # Set the reference number
    lim = config.get('reference_number_start', 1)
    ref_batches = range(lim, lim+len(config["reference_seed"]))

    disordered='disordered' if config['disordered'] else 'non-disordered'

    # Compute the reference
    print('Start computing MPO reference ({}) for {} with {} sites...'.format(
        disordered, config['hamiltonian'], config['n_sites']))
    if config['hamiltonian']=='fermi-hubbard-1d':
        for ref_batch, ref_seed in zip(ref_batches, config["reference_seed"]):
            _, T, V = construct_spinful_FH1D_hamiltonian(
                config['n_orbitals'], get_matrix=False, disordered=config['disordered'], reference_seed=ref_seed)    
            if config['compute_maximum_time']: compute_maximum_duration(config, T=-T, V=-V)
            if config['compute_reference']: compute_reference(config, ref_batch, ref_seed, T=-T, V=-V)

    elif config['hamiltonian']=='ising-1d':
        for ref_batch, ref_seed in zip(ref_batches, config["reference_seed"]):
            J, g, h = -config['J'], -config['g'], -config['h']
            _, Js, gs, hs = construct_ising_hamiltonian(config['n_sites'], J, g, h, disordered=config['disordered'], 
                                                        get_matrix=False, reference_seed=ref_seed)
            if config['compute_maximum_time']: compute_maximum_duration(config, J=Js, g=gs, h=hs)
            if config['compute_reference']: compute_reference(config, ref_batch, ref_seed, J=Js, g=gs, h=hs)

    elif config['hamiltonian']=='heisenberg':
        for ref_batch, ref_seed in zip(ref_batches, config["reference_seed"]):
            J, h = -asarray(config['J']), -asarray(config['h'])
            _, Js, hs = construct_heisenberg_hamiltonian(config['n_sites'], J, h, disordered=config['disordered'], 
                                                         get_matrix=False, reference_seed=ref_seed)
            if config['compute_maximum_time']: compute_maximum_duration(config, J=Js, h=hs)
            if config['compute_reference']: compute_reference(config, ref_batch, ref_seed, J=Js, h=hs)

    get_duration(t0)

if __name__ == "__main__":
    main()