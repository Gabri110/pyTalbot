import os
import numpy as np
import ctypes


def perform_integrals(config):
    '''This mehtod computes the integrals in (4.8) using GSL's QAG and CQUAD integration methods.
    In order to minimise the size of the integrals, we only compute integrals in the intervals [t_i-1, t_i].
    To recover the original interval [z_i, t_i] we only have to use Barrow's rule.

    Note that we must have integrands independent of t, so we use the identity 
    sin(tau-t) = cos(t)sin(tau) - sin(t)cos(tau) and separate each integral into two different ones.

    This function is accelerated through OpenMP and, optionally, MPI.


    Parameters
    ----------
    config : TalbotConfig
        The class containing all the parameters of the simulation.

    Returns
    -------
    result : np.ndarray of floats
        The values of the integrals at (n,t_i,z_i). The array has shape (N_max, N_t, N_z).
    '''

    # Start MPI
    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        use_MPI = True
    except ImportError:
        use_MPI = False
        rank = 0
        size = 1

    print(f"This is process {rank} of {size}.")

    # We store the range of n, kn and gn in arrays of length N_max
    n_values = np.linspace(0, config.N_max-1, config.N_max, dtype=int)
    k_n_values = 2 * np.pi * n_values

    # We store the range of t, z in the arrays of dimension (N_t) and (N_z)
    t_values = np.linspace(config.z_T/config.c * config.initial_t_zT, config.z_T/config.c * config.final_t_zT, config.N_t)
    z_values = np.linspace(0, config.N_z*config.delta_z, config.N_z)

    # Preallocate the result arrays (shape: len(n_values) x len(z_values))
    x_min = np.zeros((len(t_values), len(z_values)))
    x_max = np.empty((len(t_values), len(z_values)))

    # Compute the values
    t1_values = np.roll(t_values, 1)
    t1_values[0] = 0.
    np.maximum(z_values[None, :], t1_values[:, None], out=x_min)  # Max between z_values[j] and t_values[i-1]
    np.maximum(z_values[None, :], t_values[:, None], out=x_max)  # Max between z_values[j] and t_values[i]

    # Distribute loop iterations
    start = rank * np.ceil(len(n_values) / size).astype('int')
    end = (rank + 1) * np.ceil(len(n_values) / size).astype('int') if rank != size-1 else len(n_values)

    # Prealocate the result arrays. Their size depends on the rank
    partial_integral_sin_rank = np.zeros((end - start,len(t_values),len(z_values)))
    partial_integral_cos_rank = np.zeros((end - start,len(t_values),len(z_values)))


    # Load the compiled libraries with ctypes
    int_t = ctypes.c_int
    double_t = ctypes.c_double
    ptr_t = ctypes.POINTER(double_t)

    # Utility function
    ptr = lambda array: array.ctypes.data_as(ptr_t)

    # We load the compiled libraries with ctypes
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
    fast_integrals = ctypes.CDLL(os.path.join(lib_path, f'integrals.so'))

    fast_integrals.compute_integrals.argtypes = [
        ptr_t, ptr_t, ptr_t, ptr_t, ptr_t, ptr_t, ptr_t,
        int_t, int_t, int_t,
        int_t, double_t, double_t,
        double_t, int_t, int_t, double_t
    ]
    fast_integrals.compute_integrals.restype = None


    # We perform the integrals
    limit = 50_000
    eps_abs = 1e-6
    eps_rel = 1e-5
    epsilon = 5e-6

    # We compute the integrals
    fast_integrals.compute_integrals(
            ptr(partial_integral_cos_rank), ptr(partial_integral_sin_rank), ptr(x_min), ptr(x_max), 
            ptr(k_n_values), ptr(t_values), ptr(z_values), 
            len(n_values), len(t_values), len(z_values), 
            limit, eps_abs, eps_rel, 
            config.omega, start, end, epsilon
        )

    if use_MPI:
        print(f"Rank {rank}: Finished!")
        comm.Barrier()
    
        # We preparate the output arrays
        if rank == 0:
            partial_integral_sin_gathered = np.empty((size,) + partial_integral_sin_rank.shape, dtype=np.float64)
            partial_integral_cos_gathered = np.empty((size,) + partial_integral_cos_rank.shape, dtype=np.float64)
        else:
            partial_integral_sin_gathered = None
            partial_integral_cos_gathered = None

        # We gather everything in the first node
        comm.Gather(partial_integral_sin_rank, partial_integral_sin_gathered, root=0)
        comm.Gather(partial_integral_cos_rank, partial_integral_cos_gathered, root=0)


        # We no longer need the extra nodes
        if rank != 0:
            exit()
            
        resummed_integral_sin_long = np.empty((size * np.ceil(len(n_values) / size).astype('int'), len(t_values),len(z_values)), dtype=np.float64)
        resummed_integral_cos_long = np.empty((size * np.ceil(len(n_values) / size).astype('int'), len(t_values),len(z_values)), dtype=np.float64)

        # Post-process aggregated results
        np.concatenate(partial_integral_sin_gathered, out=resummed_integral_sin_long)
        np.concatenate(partial_integral_cos_gathered, out=resummed_integral_cos_long)
        del partial_integral_sin_gathered, partial_integral_cos_gathered, partial_integral_sin_rank, partial_integral_cos_rank

        # We adjust the shape
        resummed_integral_sin = resummed_integral_sin_long[:len(n_values),:,:]
        resummed_integral_cos = resummed_integral_cos_long[:len(n_values),:,:]
        del resummed_integral_sin_long, resummed_integral_cos_long
    else:
        resummed_integral_sin = partial_integral_sin_rank
        resummed_integral_cos = partial_integral_cos_rank
        del partial_integral_sin_rank, partial_integral_cos_rank



    # Cumulative sum along the 't' axis (axis=1) to recover the full integrals
    for i in range(1, resummed_integral_sin.shape[1]):
        resummed_integral_sin[:, i, :] += resummed_integral_sin[:, i-1, :]
        resummed_integral_cos[:, i, :] += resummed_integral_cos[:, i-1, :]


    # Initialize the result array
    result = np.empty((len(n_values), len(t_values), len(z_values)))
    result = np.sin(config.omega * t_values[None,:,None]) * resummed_integral_cos \
        - np.cos(config.omega * t_values[None,:,None]) * resummed_integral_sin # sin(tau-t) = cos(t)sin(tau) - sin(t)cos(tau)

    del resummed_integral_sin, resummed_integral_cos

    return result if rank == 0 else None