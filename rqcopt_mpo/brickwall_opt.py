import os

import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import vmap
from jax import config as c
c.update("jax_enable_x64", True)

from .adam import RieADAM
from .util import project_unitary_tangent, retract_unitary, inner_product
from .tn_brickwall_methods import get_riemannian_gradient_and_cost_function


def optimize_swap_network_circuit_RieADAM(config, U, Vlist_start):
    """
    Optimize the quantum gates in a swap network layout to approximate
    the unitary matrix `U` using a Riemannian ADAM optimizer.
    Vlist_start is given in the form of tensors or matrices.
    """
    
    assert(Vlist_start.shape[1:]==(2,2,2,2))
    f_df = lambda vlist: get_riemannian_gradient_and_cost_function(
        U, vlist, config['n_sites'], config['degree'], config['n_repetitions'], config['n_id_layers'], 
        config['max_bondim'], config['normalize_reference'], config['hamiltonian'])

    # Define retraction, projection, and inner product
    _retract = lambda v, eta: retract_unitary(v, eta, use_TN=True)
    _project = lambda u,z: project_unitary_tangent(u,z, True)
    retract, project = vmap(_retract), vmap(_project)
    
    # Set up the optimizer
    _metric = lambda v,x,y: inner_product(v,x,y,True)
    metric = vmap(_metric)
    opt = RieADAM(maxiter=config['n_iter'], lr=float(config['lr']))
    Vlist, neval, err_iter = opt.minimize(function=f_df, initial_point=Vlist_start,
                                          retract=retract, projection=project, metric=metric)


    err_iter1 = jnp.asarray(err_iter)
    # Cost function: Frobenius norm
    err_init = err_iter[0]
    err_opt = jnp.min(jnp.asarray(err_iter))
    err_end = err_iter[-1]

    print(f"err_init: {err_init}")
    print(f"err_end after {neval} iterations: {err_end}")
    print(f"err_opt: {err_opt}")
    print(f"err_init/err_opt: {err_init/err_opt}")

    _ = plot_loss(config, err_iter, err_opt, save=True)
                
    return Vlist, err_iter


def plot_loss(config, err_iter, err_opt, save=False):
    # Visualize optimization progress
    points = jnp.arange(len(err_iter))
    err_iter = jnp.asarray(err_iter)

    label = 'err_init={:.2e}\nerr_end={:.2e}\nerr_opt={:.2e}\nerr_init/err_opt={:.4f}'.format(
        err_iter[0], err_iter[-1], err_opt, err_iter[0]/err_opt)
    title = f"RieADAM with lr={config['lr']} for {config['n_sites']} sites, $t=${config['t']}"
    title += ', '+str(config['n_repetitions']) + ' repetitions'

    plt.figure(dpi=300)
    plt.semilogy(points, err_iter, '.-', label=label)
    plt.xlabel("Iteration")
    plt.ylabel("$\mathcal{C}$")
    plt.legend()
    plt.grid(True)
    plt.title(title)
    plt.tight_layout()
    
    if save:
        fname = str(config['model_nbr']) + '_loss.pdf'
        fdir = os.path.join(config['model_dir'], fname)
        plt.savefig(fdir)