import jax.numpy as jnp
from jax.numpy.linalg import qr
from scipy.linalg import rq
from jax import config
config.update("jax_enable_x64", True)    

def get_identity_two_qubit_tensors(n, use_TN=True):
    '''
    Generate n pairs of identity tensors that correspond to two-qubit gates.
    '''
    if use_TN: tn = jnp.asarray([jnp.eye(4).reshape((2,2,2,2)) for _ in range(n)])
    else: tn = jnp.asarray([jnp.eye(4) for _ in range(n)])
    return tn

def get_additional_layers(n_orbitals, n_id_layers):
    """
    Get n_fragment_layers of a brickwall circuit that are initialized to identity or randomly.
    We always start with an even layer.
    """
    N_odd_gates, N_even_gates = n_orbitals, n_orbitals-1  # Number of gates per layer
    n_fragment_gates = int(jnp.floor(n_id_layers/2))*N_even_gates + int(jnp.ceil(n_id_layers/2))*N_odd_gates
    fragment_gates = get_identity_two_qubit_tensors(n_fragment_gates)
    return jnp.asarray(fragment_gates)

def get_id_mpo(nsites, use_dummy_bonds=True):
    '''
    Construct an MPO with local identity tensors of nsites sites.
    '''
    TN_id_ = jnp.eye(2)
    TN_id = jnp.expand_dims(TN_id_, (0,-1))
    if not use_dummy_bonds:
        TN_id_first = jnp.expand_dims(TN_id_, -1)
        TN_id_last = jnp.expand_dims(TN_id_, 0)
        mpo = [TN_id_first]
        mpo += [TN_id for _ in range(nsites-2)]
        mpo += [TN_id_last]
    else:
        mpo = [TN_id for _ in range(nsites)]
    return mpo

def reduce_dummy_bonds(mpo):
    if len(mpo[0].shape)==(len(mpo[-2].shape)-1): pass
    elif mpo[0].shape[0]==1: mpo[0] = mpo[0][0]
    else: raise Exception('First dimension is no dummy')
    if len(mpo[-1].shape)==(len(mpo[-2].shape)-1): pass
    elif mpo[-1].shape[-1]==1: mpo[-1] = mpo[-1][...,0]
    else: raise Exception('Last dimension is no dummy')
    return mpo

def get_maximum_bond_dimension(mpo):
    dims = jnp.asarray([jnp.max(jnp.asarray(m.shape)) for m in mpo])
    max_dim = int(jnp.max(dims))
    return max_dim

def convert_mpo_to_mps(mpo):
    d = mpo[0].shape[-2]
    mps = [T.reshape((T.shape[0], d**2, T.shape[-1])) for T in mpo]
    return mps

def convert_mps_to_mpo(mps):
    d = 2  #jnp.sqrt(mps[0].shape[-2])
    mpo = [T.reshape((T.shape[0], d, d, T.shape[-1])) for T in mps]
    return mpo

def get_mpo_from_matrix(U):
    '''
    Decompose a full-rank matrix into an MPO by means of SVD.
    '''
    
    tensor_list = []
    n = int(jnp.round(jnp.log2(jnp.shape(U)[0])))

    shape = (2,2,)*n
    A = U.reshape(shape)

    for site in range(1, n):
        n_ = n-site+1  # Particles still left to SVD
        if site==1: A_perm = jnp.moveaxis(A, n_, 1)
        elif 1<site<n: A_perm = jnp.moveaxis(A, n_+1, 2)

        shape_1 = A_perm.shape
        lim=2 if site==1 else 3  # Exception for first round
        shape_2 = (int(jnp.prod(jnp.asarray(shape_1[:lim]))),
                   int(jnp.prod(jnp.asarray(shape_1[lim:]))))
        
        B = A_perm.reshape(shape_2)

        u,s,v = jnp.linalg.svd(B, full_matrices=False)
        
        D = jnp.diag(s)@v
        shape_3 = shape_1[:lim] + (u.shape[-1],)
        
        E = u.reshape(shape_3)
        tensor_list.append(E)
        shape_4 = (u.shape[-1],) + shape_1[lim:]
        F = D.reshape(shape_4)
        if site==n-1: tensor_list.append(F)
        A = F.copy()

    # Add dummy legs
    tensor_list[0] = tensor_list[0][jnp.newaxis,...]
    tensor_list[-1] = tensor_list[-1][...,jnp.newaxis]

    return tensor_list

def get_matrix_from_mpo(tensor_list):
    '''
    This function contracts an MPO in a linear way.
    '''
    A = tensor_list[0]
    for i in range(len(tensor_list)-1):
        B = tensor_list[i+1]        
        C = jnp.einsum('iabj,jcdk->iacbdk', A, B)
        shape = C.shape
        n,m,l,o = shape[0], shape[1]*shape[2], shape[3]*shape[4], shape[-1]
        A = C.reshape((n,m,l,o)).copy()
    A = jnp.einsum('iabj->ab', A)
    return A

def compress_SVD(u, s, v, max_bondim=128):
    '''
    Compress an MPO to a maximum bond dimension max_bondim.
    '''
    if max_bondim>=len(s): pass
    else:
        u = u[..., :max_bondim]
        v = v[:max_bondim, ...]
        s = s[:max_bondim]
    return u, s, v

def split_tensor_into_half_canonical_mpo_pair(T, canonical_mode='left', max_bondim=128):
    """
          |                 |
       __|‾|__            --O--
         | |      -->       |
       ‾‾|_|‾‾            --O--
          |                 |
    * Left canonical: mpo1=U, mpo2=S@V
    * Right canonical: mpo1 = U@S, mpo2=V
    """
    lim=3; ind1=3; ind2=2; assert len(T.shape)==6

    # Permute axis
    A = jnp.moveaxis(T, ind1, ind2)
    shape_1 = A.shape
    # Reshape to matrix
    shape_2 = (int(jnp.prod(jnp.asarray(shape_1[:lim]))),
                    int(jnp.prod(jnp.asarray(shape_1[lim:]))))
    B = A.reshape(shape_2)
    
    # Perform SVD
    u,s,v = jnp.linalg.svd(B, full_matrices=False)
    if False not in [jnp.isfinite(u).all(), jnp.isfinite(s).all(), jnp.isfinite(v).all()]:
        u, s, v = compress_SVD(u, s, v, max_bondim)

        if canonical_mode=='left': mpo1=u; mpo2=jnp.diag(s)@v
        elif canonical_mode=='right': mpo1=u@jnp.diag(s); mpo2=v
        else: raise Exception('canonical_mode has to be left or right')
    
    else:
        if canonical_mode=='right': mpo1, mpo2 = rq(B, mode='economic')
        elif canonical_mode=='left': mpo1, mpo2 = jnp.linalg.qr(B, mode='reduced')    

    # Get first MPO
    shape_3 = shape_1[:lim] + (mpo1.shape[-1],)
    E = mpo1.reshape(shape_3)
    # Get second MPO
    shape_4 = (mpo2.shape[0],) + shape_1[lim:]
    F = mpo2.reshape(shape_4)

    return E, F

def split_tensor_into_mpo(T, boundary=4, method='QR', max_bondim=128):
    '''
    This function takes a tensor T acting on two qubits
    and splits it by means of QR/SVD into an MPO.

    boundary=2:
          |                 |
       __|‾|__            --O--
         | |      -->       |
       ‾‾|_|‾‾            --O--
          |                 |
    
    Merge these two tensors again: 
    boundary=1: jnp.einsum('abc,cdef->adbef', E, F)
    boundary=2: jnp.einsum('iabc,cdef->iadbef', E, F)
    boundary=3: jnp.einsum('iabc,cde->iadbe', E, F)

    boundary=4:
                        
       __|‾|__            --O--
         | |      -->       |
       ‾‾|_|‾‾            --O--
    '''
 
    if boundary==1: lim=2; ind1=2; ind2=1; assert len(T.shape)==5
    elif boundary==2: lim=3; ind1=3; ind2=2; assert len(T.shape)==6
    elif boundary==3: lim=3; ind1=3; ind2=2; assert len(T.shape)==5
    elif boundary==4: lim=2; ind1=2; ind2=1; assert len(T.shape)==4

    # Permute axis
    A = jnp.moveaxis(T, ind1, ind2)
    shape_1 = A.shape
    # Reshape to matrix
    shape_2 = (int(jnp.prod(jnp.asarray(shape_1[:lim]))),
                    int(jnp.prod(jnp.asarray(shape_1[lim:]))))
    B = A.reshape(shape_2)
    
    if method=='SVD':
        # Perform SVD
        u,s,v = jnp.linalg.svd(B, full_matrices=False)
        u, s, v = compress_SVD(u, s, v, max_bondim)
        D = jnp.diag(s)@v
    elif method=='QR':
        # Perform QR
        u,D = jnp.linalg.qr(B, mode='reduced')
    # Get first MPO
    shape_3 = shape_1[:lim] + (u.shape[-1],)
    E = u.reshape(shape_3)
    # Get second MPO
    shape_4 = (D.shape[0],) + shape_1[lim:]
    F = D.reshape(shape_4)

    return E, F

def merge_two_mpos_and_gate(V, mpo1, mpo2, gate_is_left=True):
    '''
    This function does not split the two-qubit gates but rather contracts
    directly.

    gate_is_left=True for two-qubit gate on left side of MPO, False 
    if on right side of MPO.

    Merge the following:
             |  
    ---|‾|---O--- 
       | |   |     
    ---|_|---O--- 
             |
    '''
    if gate_is_left:
        contraction_path = 'iabc,cdef,begh->iadghf'
        A = jnp.einsum(contraction_path, mpo1, mpo2, V)
    else:
        contraction_path = 'abcd,icef,fdgh->iabegh'
        A = jnp.einsum(contraction_path, V, mpo1, mpo2)          

    return A

def merge_mpo_and_layer(gates_in_layer, mpo_init, layer_is_odd, layer_is_left=True):
    mpo_res=[]; n_spin_orbitals=len(mpo_init)

    # Merge each gate/MPO in the layer
    i_mpo_init = 0
    if not layer_is_odd: mpo_res.append(mpo_init[0]); i_mpo_init+=1
    while i_mpo_init+1<n_spin_orbitals:
        mpo1, mpo2 = mpo_init[i_mpo_init:i_mpo_init+2]  # Pair of MPO
        gate = gates_in_layer[int(i_mpo_init/2)]  # Gate to be merged
        merged_T = merge_two_mpos_and_gate(gate, mpo1, mpo2, layer_is_left)
        # Split the current tensors
        T1, T2 = split_tensor_into_mpo(merged_T, boundary=2)
        mpo_res.append(T1); mpo_res.append(T2)
        i_mpo_init += 2
    if i_mpo_init==n_spin_orbitals-1: mpo_res += [mpo_init[-1]]

    return mpo_res

def fully_contract_mpo(mpo):
    '''
    Fully contract an MPO by obtaining its trace.
    '''
    # First contract the physical bonds
    for i, T in enumerate(mpo):
        assert len(T.shape)==4
        mpo[i] = jnp.einsum('iaaj->ij', T)
    # Now contract the inner bonds
    T = mpo[0]
    for i in range(1, len(mpo)):
        T_next = mpo[i]
        T = jnp.einsum('ij,jk->ik', T, T_next)
    trace = jnp.einsum('ii->i', T)[-1]
    return trace

def left_to_right_QR_sweep(mpo, get_norm=False, normalize=False):
    mpo_res = mpo.copy()
    for i in range(0, len(mpo_res)-1):  # Go from left to right
        mpo = mpo_res[i]  # Current MPO
        shape = mpo.shape  # (l,2,2,r)
        mpo = mpo.reshape((-1, shape[-1]))  # Matrix form
        Q, R = jnp.linalg.qr(mpo, mode='reduced')
        mpo_res[i] = Q.reshape(shape[:-1]+ (Q.shape[-1],))
        merged_mpo = jnp.einsum('ab,bcde->acde', R, mpo_res[i+1])
        mpo_res[i+1] = merged_mpo.reshape((R.shape[0],)+mpo_res[i+1].shape[1:])

    mpo = mpo_res[-1]
    shape = mpo.shape
    mpo = mpo.reshape((-1,shape[-1]))
    Q, R = jnp.linalg.qr(mpo, mode='reduced')

    # Get norm
    nrm = R.reshape(-1)[0]
    if nrm<0: nrm=-nrm; R,Q=-R,-Q  # Flip sign of last tensor and dummy (norm) tensor to ensure positive norm
    # Normalize
    if normalize: mpo_res[-1] = Q.reshape(shape[:-1]+(Q.shape[-1],))
    else: mpo_res[-1] = Q.reshape(shape[:-1]+(Q.shape[-1],)) * R.reshape(-1)[0]  # Absorb normalization
    
    if get_norm: return mpo_res, nrm.real
    else: return mpo_res

def right_to_left_RQ_sweep(mpo_res, get_norm=False, normalize=False):
    for i in reversed(range(1, len(mpo_res))):  # Go from right to left
        mpo = mpo_res[i]  # Current MPO
        shape = mpo.shape  # (l,2,2,r)
        mpo = mpo.reshape((shape[0], -1))  # Matrix form
        R, Q = rq(mpo, mode='economic')
        mpo_res[i] = Q.reshape((Q.shape[0],)+ shape[1:])
        merged_mpo = jnp.einsum('abcd,de->abce', mpo_res[i-1], R)
        mpo_res[i-1] = merged_mpo.reshape(mpo_res[i-1].shape[:-1]+ (R.shape[-1],))

    mpo = mpo_res[0]
    shape = mpo.shape
    mpo = mpo.reshape((mpo.shape[0], -1))
    R, Q = rq(mpo, mode='economic')

    # Norm
    nrm = R.reshape(-1)[0]
    if nrm<0: nrm=-nrm; R,Q=-R,-Q  # Flip sign of last tensor and dummy (norm) tensor to ensure positive norm
    # Normalize
    if normalize: mpo_res[0] = Q.reshape((Q.shape[0],)+ shape[1:])
    else: mpo_res[0] = Q.reshape((Q.shape[0],)+ shape[1:]) * R.reshape(-1)[0]  # Absorb normalization
    
    if get_norm: return mpo_res, nrm.real
    else: return mpo_res

def left_to_right_SVD_sweep(mpo_res, max_bondim=128):
    # Right-to-left sweep to compress MPO via SVD
    for i in range(len(mpo_res)-1):
        mpo = mpo_res[i]
        shape = mpo.shape
        mpo = mpo.reshape((-1, shape[-1]))
        u, s, v = jnp.linalg.svd(mpo, full_matrices=False)
        u, s, v = compress_SVD(u, s, v, max_bondim)
        mpo_res[i] = u.reshape(shape[:-1]+(u.shape[-1],))
        SV = jnp.diag(s)@v
        mpo_res[i+1] = jnp.einsum('ab,bcde->acde', SV, mpo_res[i+1])

    mpo = mpo_res[-1]
    shape = mpo.shape
    mpo = mpo.reshape((-1, shape[-1]))
    u, s, v = jnp.linalg.svd(mpo, full_matrices=False)
    u, s, v = compress_SVD(u, s, v, max_bondim)
    U = u.reshape(shape[:-1]+(u.shape[-1],))
    SV = jnp.diag(s)@v
    mpo_res[-1] = U * SV.reshape(-1)[0]  # Absorb normalization

    return mpo_res

def merge_mpo(mpo1, mpo2, two_qubit_gate=True, T1=None):
    '''
    This function is used for obtaining the MPO reference.
    This function merges a two-qubit gate in
    MPO form with the MPOs in the next layer.
    two_qubit_gate=True: One of the MPOs is a two-qubit gate
    Yu's approach

       |   |                    |
    ---0---O---             ---XXX---
       |   |       -->          |
    ---0---O---             ---XXX---
       |   |                    |

    '''
    if two_qubit_gate: contr_path1 = 'abc,fbgh->faghc'; contr_path2 = 'cde,heij->hcdij'
    else: contr_path2 = 'dhij,gikl->gdhklj'
    if type(T1)==type(None): T1 = jnp.einsum(contr_path1, mpo2[0], mpo1[0])
    else: pass
    T1 = T1.reshape(T1.shape[:-2]+(-1,))  # Double bond -> one bond
    T2 = jnp.einsum(contr_path2, mpo2[1], mpo1[1])
    T2 = T2.reshape((-1,)+T2.shape[2:])  # Double bond -> one bond

    return T1, T2

def compress_mpo(mpo_init, max_bondim=128):
    # Compress the MPO
    mpo = mpo_init.copy()
    mpo = right_to_left_RQ_sweep(mpo)
    mpo = left_to_right_SVD_sweep(mpo, max_bondim)
    return mpo
    
def get_right_canonical_mps(mps):
    """
    Converts a list of local tensors (MPS) into right canonical form using RQ decomposition.
    """
    updated_mps = mps.copy()
    right_canonical_mps = [None]*len(updated_mps)
    
    for i in reversed(range(1,len(updated_mps))):
        # Reshape the tensor into a matrix for RQ decomposition
        tensor = updated_mps[i]
        shape = tensor.shape
        tensor = tensor.reshape(shape[0], -1)
        # Perform RQ decomposition
        R, Q = rq(tensor, mode='economic')
        # Update the current tensor to be Q
        right_canonical_mps[i] = Q.reshape(-1, shape[1], shape[2])
        # Multiply R with the previous tensor
        updated_mps[i-1] = jnp.einsum('ijk,kl->ijl', updated_mps[i - 1], R)
    
    # Set the first tensor
    right_canonical_mps[0] = updated_mps[0]
    
    return right_canonical_mps

def get_left_canonical_mps(mps, normalize=False, get_norm=False):
    """
    Converts a list of local tensors (MPS) into left canonical form using QR decomposition.
    """
    left_canonical_mps, updated_mps = [], mps.copy()
    for i in range(len(updated_mps)-1):
        # Reshape the tensor into a matrix for QR decomposition
        tensor = updated_mps[i]
        shape = tensor.shape
        tensor = tensor.reshape(shape[0]*shape[1],shape[-1])
        # Perform QR decomposition
        Q, R = jnp.linalg.qr(tensor, mode='reduced')
        # Update the current tensor to be Q
        left_canonical_mps.append(Q.reshape(shape[0],shape[1],-1))
        # Multiply R with the next tensor
        updated_mps[i+1] = jnp.einsum('ij,jkl->ikl', R, updated_mps[i+1])
    
    # Append the last tensor
    if normalize or get_norm:
        tensor = updated_mps[-1]
        shape = tensor.shape
        tensor = tensor.reshape((-1,shape[-1]))
        Q, R = jnp.linalg.qr(tensor, mode='reduced')

        # Get norm
        nrm = R.reshape(-1)[0]
        if nrm<0: nrm=-nrm
        # Normalize
        if normalize: left_canonical_mps.append(Q.reshape(shape[:-1]+(Q.shape[-1],)))
        else: left_canonical_mps.append(updated_mps[-1])
        if get_norm: return left_canonical_mps, nrm.real
    else:
        left_canonical_mps.append(updated_mps[-1])
    
    return left_canonical_mps

def inner_product_mps(mps1, mps2):
    """
    Compute the inner product between two MPS by merging the local tensors using numpy.einsum.
    """
    # Initialize the contraction result as an identity matrix
    contraction_result = jnp.eye(mps1[0].shape[0])
    # Iterate over the local tensors and merge them using einsum
    for tensor1, tensor2 in zip(mps1, mps2):
        contraction_result = jnp.einsum('ij,ikl,jkm->lm', contraction_result, tensor1, jnp.conjugate(tensor2))
    # Return the trace of the final contraction result
    return jnp.trace(contraction_result)

def canonicalize_local_tensor(T, left=True):
    """
    Method to left-orthogonalize (left=True) or right-orthogonalize 
    (left=False) a local tensor of an MPO.
    """
    shape = T.shape
    if left:
        T = T.reshape((-1, shape[-1]))  # Matrix form
        Q, R = qr(T, mode='reduced')
        T_can = Q.reshape(shape[:-1]+(Q.shape[-1],))
    else:
        T = T.reshape((shape[0], -1))  # Matrix form
        R, Q = rq(T, mode='economic')
        T_can = Q.reshape((Q.shape[0],)+ shape[1:])
    return T_can, R