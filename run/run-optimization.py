import os
from sys import argv
from yaml import safe_load
from time import time
from threading import Thread

from jax.numpy import asarray
from jax import config as c
c.update("jax_enable_x64", True)

from rqcopt_mpo.save_model import (save_optimized_model, remove_blank_lines, 
                                   get_model_nbr, save_config, load_reference)
from rqcopt_mpo.spin_systems import construct_ising_hamiltonian, construct_heisenberg_hamiltonian
from rqcopt_mpo.fermionic_systems import load_molecular_model
from rqcopt_mpo.brickwall_circuit import get_nlayers, get_initial_gates
from rqcopt_mpo.brickwall_opt import optimize_swap_network_circuit_RieADAM
from rqcopt_mpo.tn_helpers import left_to_right_QR_sweep

from helpers import periodic_clear, get_duration, get_memory_usage


def molecular_dynamics_opt(config, Vlist_start, U_ref):
    # Left canonicalize the reference
    U_ref = left_to_right_QR_sweep(U_ref, get_norm=False, normalize=config['normalize_reference'])

    # Riemannian optimization
    t_start = time()
    Vlist, err_iter = optimize_swap_network_circuit_RieADAM(config, U_ref, Vlist_start)
    err_iter = asarray(err_iter).reshape(-1)
    get_duration(t_start, program='optimization') 

    # Save optimization results
    _ = save_optimized_model(config, U_ref, Vlist, err_iter, config['ref_nbr'])


def set_up_model(config, path):
    # Set script path
    config['script_path'] = path
    hamiltonian_path = os.path.join(path, config['hamiltonian'])

    # Set the current model number
    config['server'] = 'local'
    fname = 'model_list.txt' 
    model_list = os.path.join(hamiltonian_path, fname)
    remove_blank_lines(model_list)
    model_nbr = get_model_nbr(model_list)
    config['model_nbr'] = model_nbr

    # Set the directory for loading results (in case)
    if config['load']:
        load_model_dir = os.path.join(hamiltonian_path, 'results', str(config['load_which']))
        config['load_dir'] = load_model_dir
        if not os.path.isdir(load_model_dir): os.makedirs(load_model_dir)
        
    # Set the directory for saving results
    model_dir = os.path.join(hamiltonian_path, 'results', str(model_nbr))
    config['model_dir'] = model_dir
    if not os.path.isdir(model_dir): os.makedirs(model_dir)

    # Set the reference directory
    config['reference_dir'] = os.path.join(hamiltonian_path, 'reference')    

    # Set some parameters
    if 'n_id_layers' not in config.keys(): config['n_id_layers']=0
    if 'n_sites' not in config.keys(): 
        if config['hamiltonian']=='molecular': config['n_sites']=config['n_orbitals']
        elif config['hamiltonian']=='fermi-hubbard-1d': config['n_sites']=2*config['n_orbitals']
    
    return config, model_nbr



def main():
    t0 = time()
    # Start a separate thread to clear JIT caches every 15 minutes
    clear_thread = Thread(target=periodic_clear, args=(1800,))
    clear_thread.daemon = True
    clear_thread.start()
    
    # Load the config file and set up model
    with open(argv[1], 'r') as f:
        config = safe_load(f)
    path = os.getcwd()
    config, model_nbr = set_up_model(config, path)
    get_memory_usage()
    
    print('\n##### Simulation for model {} #####\n'.format(model_nbr))
    save_config(config, status='before_training')
    print('System with ... \n\t*{} Hamiltonian \n\t*n_sites = {}\n\t*degree = {} \n\t*n_repetitions = {}\n\t*n_id_layers={}'.format(
        config['hamiltonian'], config['n_sites'], config['degree'], config['n_repetitions'], config['n_id_layers']))


    # Load the references and obtain initial Trotter gates; use negative coefficients for adjoint reference
    if config['hamiltonian'] in ['molecular', 'fermi-hubbard-1d']:
        if config['hamiltonian']=='molecular':
            n_orbitals, t, _, T, V, _, U_ref, _, _ = load_molecular_model(
                config['reference_dir'], config['ref_nbr'], config['molecule'])
            config['t'] = t
        else:
            U_ref, t, _, _, _, _, _, _, _, T, V = load_reference(
                config['reference_dir'], int(config['n_sites']/2), config['ref_nbr'])  # Old convention for naming reference
        Vlist_start = get_initial_gates(config['n_sites'], config['t'], config['n_repetitions'], config['degree'], 
                                        config['hamiltonian'], config['n_id_layers'], use_TN=True, T=-T, V=-V)

    elif config['hamiltonian']=='ising-1d':
        U_ref, t, _, _, _, _, _, _, _, J, g, h = load_reference(
            config['reference_dir'], config['n_sites'], config['ref_nbr'])
        if type(J)==int:
            _, J, g, h = construct_ising_hamiltonian(config['n_sites'], J, g, h, disordered=False, get_matrix=False)
        Vlist_start = get_initial_gates(config['n_sites'], config['t'], config['n_repetitions'], config['degree'], 
                                        config['hamiltonian'], config['n_id_layers'], use_TN=True, J=-J, g=-g, h=-h)

    elif config['hamiltonian']=='heisenberg':
        U_ref, t, _, _, _, _, _, _, _, J, h = load_reference(
            config['reference_dir'], config['n_sites'], config['ref_nbr'])
        if len(J.shape)!=3:
            _, J, h = construct_heisenberg_hamiltonian(config['n_sites'], J, h, disordered=False, get_matrix=False)
        Vlist_start = get_initial_gates(config['n_sites'], config['t'], config['n_repetitions'], config['degree'], 
                                        config['hamiltonian'], config['n_id_layers'], use_TN=True, J=-J, h=-h)
        
    assert config['t']==t, "t={} for model but t={} for reference".format(config['t'], t) 

    # Obtain information about the brickwall layout
    n_orbitals_ = None if config['hamiltonian'] in ['fermi-hubbard-1d', 'ising-1d', 'heisenberg'] else n_orbitals
    n_layers = get_nlayers(config['degree'], config['n_repetitions'], n_orbitals=n_orbitals_, hamiltonian=config['hamiltonian'])
    print('Swap network with ...\n\t*n_layers = {}\n\t*n_gates = {}'.format(n_layers, len(Vlist_start)))

    print(f'\n### Run model for t={t} ###')
    molecular_dynamics_opt(config, Vlist_start, U_ref)
    
    print(f'\n##### Simulation finished! #####')
    save_config(config, status='after_training')

    get_duration(t0, program='script')
    get_memory_usage()


if __name__ == "__main__":
    main()
