import os
import pickle
from yaml import safe_load
from threading import Thread
from psutil import Process
from sys import argv

import matplotlib.pyplot as plt
import jax.numpy as jnp

from jax import config
config.update("jax_enable_x64", True)

from rqcopt_mpo import *
from rqcopt_mpo.spin_systems import *
from rqcopt_mpo.brickwall_circuit import *


def cost(Vlist_TN, U_mpo, n_sites, degree, n_repetitions, n_layers, truncation_tol, hamiltonian):
    scaling_F = int(n_sites/2)
    # Normalize the reference
    mpo = tn_helpers.left_to_right_QR_sweep(U_mpo, get_norm=False, normalize=True)
    tr = tn_brickwall_methods.fully_contract_swap_network_mpo(
        Vlist_TN, mpo, degree, n_repetitions, 0, n_layers, truncation_tol, hamiltonian)
    cost = jnp.sqrt(2 - 2*tr.real/2**scaling_F)  # Frobenius norm
    return cost

def main():
    # Load the config file
    with open(argv[1], 'r') as f:
        config = safe_load(f)
        
    # Load the reference
    current_path=os.getcwd()
    path = os.path.abspath(os.path.join(current_path, os.pardir))
    ref_path = os.path.join(path, 'run', config['hamiltonian'], 'reference')

    errors_trotter = []
    ts_trotter = []
    x_list, fx_list, popt_list = [], [], []

    for degree, n_rep_start, n_rep_end in zip(config['degree'], config['n_repetitions_start'], config['n_repetitions_end']):

        if config['hamiltonian']=='ising-1d':  
            U_mpo, t_ref, n_sites, _, _, _, _, _, _, J, g, h = save_model.load_reference(
                ref_path, config['n_sites'], config['ref_nbr'])
            Js, gs, hs = -J, -g, -h  # Get the negative coefficients (to have an adjoint reference)
            #_, Js, gs, hs = construct_ising_hamiltonian(n_sites, J, g, h)
        elif config['hamiltonian']=='heisenberg':
            U_mpo, t_ref, n_sites, _, _, _, hamiltonian, H, ref_seed, Js, hs = save_model.load_reference(
                ref_path, config['n_sites'], config['ref_nbr'])
            Js, hs = -Js, -hs  # Get the negative coefficients (to have an adjoint reference)
            #_, Js, hs = construct_heisenberg_hamiltonian(n_sites, J, h)
        elif config['hamiltonian']=='fermi-hubbard-1d':
            config['n_orbitals'] = int(config['n_sites']/2)
            U_mpo, t_ref, n_sites, _, _, _, hamiltonian, H, ref_seed, T, V = save_model.load_reference(
                ref_path, config['n_orbitals'], config['ref_nbr'])
            T, V = -T, -V  # Get the negative coefficients (to have an adjoint reference)
        

        n_repetitions = jnp.arange(n_rep_start,n_rep_end)
        errorsF_I = []

        for n in n_repetitions:
            n_layers = brickwall_circuit.get_nlayers(degree, n, hamiltonian=config['hamiltonian'])
            
            if config['hamiltonian']=='ising-1d':  
                gates = get_initial_gates(config['n_sites'], t_ref, n_repetitions=n, degree=degree, 
                                hamiltonian=config['hamiltonian'], use_TN=True, J=Js, g=gs, h=hs)    
            elif config['hamiltonian']=='heisenberg':  
                gates = get_initial_gates(config['n_sites'], t_ref, n_repetitions=n, degree=degree, 
                                hamiltonian=config['hamiltonian'], use_TN=True, J=Js, h=hs)    
            elif config['hamiltonian']=='fermi-hubbard-1d':  
                gates = get_initial_gates(config['n_sites'], t_ref, n_repetitions=n, degree=degree, 
                                hamiltonian=config['hamiltonian'], use_TN=True, T=T, V=V)  
            err = cost(gates, U_mpo, config['n_sites'], degree, n, n_layers, 
                       config['truncation_tol'], config['hamiltonian'])
            errorsF_I.append(err)
        
        

        ts = t_ref/n_repetitions
        if config['hamiltonian']=='ising-1d': lim=1
        else: lim=0
        x1, fx1, popt1 = util.get_trotter_scaling(ts[lim:], errorsF_I[lim:])

        errors_trotter.append(errorsF_I)
        ts_trotter.append(ts)
        x_list.append(x1)
        fx_list.append(fx1)
        popt_list.append(popt1)


    fig, ax = plt.subplots(1,1, figsize=(5, 3))
    colors = ['b','r','g']
    for ts_, errs_, x, fx, popt, c in zip(ts_trotter, errors_trotter, x_list, fx_list, popt_list, colors):
        ax.plot(ts_, errs_, c+'o-', label='Fit: $f(x)=a\cdot x^b$; a={}, b={}'.format(
            jnp.round(jnp.exp(popt[0]),2), jnp.round(popt[1], 2)))
        ax.plot(x, fx(x), 'k-')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('Duration of Trotter time steps')
    ax.set_ylabel('Frobenius norm')
    ax.set_title('FSG time evolution vs. exact for {} sites (order {})'.format(n_sites, config['degree']))
    ax.legend()
    fig.tight_layout()
    path = os.getcwd()
    fname = os.path.join(path, str(n_sites)+'_'+config['hamiltonian']+'_trotter'+'.pdf')
    plt.savefig(fname)
    plt.show()



if __name__ == "__main__":
    main()