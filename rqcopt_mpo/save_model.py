import os
from pickle import load as pickle_load
from pickle import dump
from numpy import save


def load_reference(path, N, ref_nbr=None):
    fname = str(N)+'_reference'
    if type(ref_nbr) is not type(None): fname += '_'+str(ref_nbr)
    fname += '.pkl'
    with open(os.path.join(path, fname), 'rb') as fp:
        reference = pickle_load(fp)
        U_mpo = reference['U_mpo']
        T = reference.get('T', None)
        V = reference.get('V', None)
        J = reference.get('J', None)
        g = reference.get('g', None)
        h = reference.get('h', None)
        t = reference['t']
        n_orbitals = reference.get('n_orbitals', None)
        n_sites = reference.get('n_sites', None)
        degree = reference['degree']
        n_repetitions = reference['n_repetitions']
        H = reference['H']
        hamiltonian = reference.get('hamiltonian', None)
        Hamiltonian = reference.get('Hamiltonian', None)
        err_threshold = reference["err_threshold"]
        ref_seed = reference["ref_seed"]

        if type(n_sites) is type(None) and type(n_orbitals) is not type(None):
            if 'fermi-hubbard-1d' in path: n_sites=2*n_orbitals

    if 'fermi-hubbard-1d' in path:
        return U_mpo, t, n_sites, degree, n_repetitions, err_threshold, Hamiltonian, H, ref_seed, T, V
    elif 'ising-1d' in path:
        return U_mpo, t, n_sites, degree, n_repetitions, err_threshold, hamiltonian, H, ref_seed, J, g, h 
    elif 'heisenberg' in path:
        return U_mpo, t, n_sites, degree, n_repetitions, err_threshold, hamiltonian, H, ref_seed, J, h


def save_reference(path, U_mpo, t, n_sites, degree, n_repetitions, 
                   err_threshold=None, hamiltonian=None, 
                   H=None, ref_seed=None, ref_nbr=None, **kwargs):
    """
    Hamiltonian: type of system, e.g., FH1d, molecular, ...
    H: numerical Hamiltonian
    """
    if not os.path.exists(path):
        # Create a new directory because it does not exist
        os.makedirs(path)
    res_opt = {'U_mpo': U_mpo, 't': t, 'n_sites': n_sites, 
    'degree': degree, 'n_repetitions': n_repetitions, 
    'H': H, "hamiltonian": hamiltonian, "err_threshold": err_threshold, "ref_seed": ref_seed, "ref_nbr": ref_nbr}
    if hamiltonian=='fermi-hubbard-1d': 
        res_opt['T'] = kwargs['T']
        res_opt['V'] = kwargs['V']
        N = int(n_sites/2)
    else:
        N = n_sites
    if hamiltonian in ['ising-1d', 'heisenberg']: 
        res_opt['J'] = kwargs['J']
        res_opt['h'] = kwargs['h']
    if hamiltonian=='ising-1d':
        res_opt['g'] = kwargs['g']
    if type(ref_nbr) is not type(None):
        fname = os.path.join(path, str(N)+'_reference_'+str(ref_nbr)+'.pkl')
    else:
        fname = os.path.join(path, str(N)+'_reference'+'.pkl')          
    with open(fname, 'wb') as fp:
        dump(res_opt, fp)


def save_optimized_model(config, U_ref, Vlist, err_iter_F, ref_nbr):
    model_nbr = str(config['model_nbr'])

    # Errors: Frobenius norm
    err_init_F, err_opt_F = err_iter_F[0], min(err_iter_F), 
    err_rel_F = err_init_F/err_opt_F

    res_opt = {'U_ref': U_ref, 'Vlist': Vlist, 'err_iter': err_iter_F, 'err_rel': err_rel_F,
            'err_init': err_init_F, 'err_opt': err_opt_F}
    with open(os.path.join(config['model_dir'], model_nbr+'_'+str(ref_nbr)+'_errors.pkl'), 'wb') as fp:
        dump(res_opt, fp)

    # Try to delete prior stored reference
    try: os.remove(os.path.join(config['model_dir'], model_nbr+'_U.pkl'))
    except: pass

        

def save_config(config, status='before_training'):
    # status either 'before_training' or 'after_training' or 'intermediate'

    # Store intermediate config
    if status=='intermediate':
        filename = '{}_config.npy'.format(config['model_nbr'])
        fdir = os.path.join(config['model_dir'], filename)
        _ = save(fdir, config)

    elif status in ['before_training', 'after_training']:
        filename = '{}_config_{}.txt'.format(config['model_nbr'], status)
        fdir = os.path.join(config['model_dir'], filename)
        with open(fdir, 'w') as f:
            if status=='before_training':
                f.write('# Configuration of model before training\n\n')
            elif status=='after_training':
                f.write('# Configuration of model after training\n\n')
            
            for key in config.keys():
                f.write('{}: {}\n'.format(key, config[key]))
                
        if status=='before_training':
            print('\n... Configuration before training saved to:\n{}\n'.format(fdir))
        if status=='after_training':
            filename = '{}_config.npy'.format(config['model_nbr'])
            fdir = os.path.join(config['model_dir'], filename)
            _ = save(fdir, config)
            print('\n... Configuration after training saved to:\n{}\n'.format(fdir))
            
            
def remove_blank_lines(fdir):
    result = ""
    with open(fdir, "r") as file:
        result += "".join(line for line in file if not line.isspace())
        result = os.linesep.join([s for s in result.splitlines() if s])
    with open(fdir, "w") as file:
        file.seek(0)
        file.write(result)
            
def get_model_nbr(model_list):
    with open(model_list, "r") as file:
        for last_line in file:
            model_nbr=int(last_line)+1
        file.close()
        print(model_nbr)
    with open(model_list, "a") as file:
        file.write('\n'+str(model_nbr))
        file.close()
    return model_nbr

    