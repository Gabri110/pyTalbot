import os
import numpy as np
import ctypes
from tqdm import tqdm



def g_n_rect_delta(n, config):
    '''Computes the g_n of the Rect function NORMALISED so that it tends to the delta function.
    For n =/= 0 we return g_n + g_{-n} = 2 g_n
    
    Parameters
    ----------
    n : np.ndarray of ints
        The ns for which to cumpute g_n. Must be integers.

    config : TalbotConfig
        The class containing all the parameters of the simulation.
    
    Returns
    -------
    g_n : np.ndarray of floats
        The values of the g_ns relative to each n.
    '''

    original_settings = np.seterr()
    np.seterr(divide='ignore', invalid='ignore')
    result = np.where(n == 0, 1, 2*np.sin(n * np.pi * config.w) / (np.pi * n * config.w**2)) # We multiply by 2 to account at the same time for g_n and g_(-n)
    np.seterr(**original_settings)
    return result


def perform_integrals(config):
    '''This mehtod computes the integrals in (4.8) using GSL's integration method.
    In order to minimise the size of the integrals, we only compute integrals in the intervals [t_i-1, t_i].
    To recover the original interval [z_i, t_i] we only have to use Barrow's rule.

    Note that we must have integrands independent of t, so we use the identity 
    sin(tau-t) = cos(t)sin(tau) - sin(t)cos(tau) and separate each integral into two different ones.

    Credits to Jérôme Richard, who explained in his stackoverflow's answer stackoverflow.com/a/79360271/24208929 
    how to call the GSL library from Python for the efficient evaluation of integrals.

    Parameters
    ----------
    config : TalbotConfig
        The class containing all the parameters of the simulation.

    Returns
    -------
    result : np.ndarray of floats
        The values of the integrals at (n,t_i,z_i). The array has shape (N_max, N_t, N_z).
    '''

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
    x_min = np.maximum(z_values[None, :], t1_values[:, None])  # Max between z_values[j] and t_values[i-1]
    x_max = np.maximum(z_values[None, :], t_values[:, None])  # Max between z_values[j] and t_values[i]
        
    # Prealocate the result arrays
    partial_integral_sin = np.zeros((len(n_values),len(t_values),len(z_values)))
    partial_integral_cos = np.zeros((len(n_values),len(t_values),len(z_values)))


    # Load the compiled libraries with ctypes
    int_t = ctypes.c_int
    double_t = ctypes.c_double
    ptr_t = ctypes.POINTER(double_t)

    # Utility function
    ptr = lambda array: array.ctypes.data_as(ptr_t)

    # We load the compiled libraries with ctypes
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib')
    fast_integrals = ctypes.CDLL(os.path.join(lib_path, 'libintegrals.so'))
    fast_integrals.compute_cos.argtypes = [
        ptr_t, ptr_t, ptr_t, ptr_t, ptr_t, ptr_t,
        int_t, int_t, int_t, int_t,
        double_t, double_t, double_t, double_t
    ]
    fast_integrals.compute_cos.restype = None

    fast_integrals.compute_sin.argtypes = [
        ptr_t, ptr_t, ptr_t, ptr_t, ptr_t, ptr_t,
        int_t, int_t, int_t, int_t,
        double_t, double_t, double_t, double_t
    ]
    fast_integrals.compute_sin.restype = None

    # We perform the integrals
    limit = 1_000_000
    eps_abs = 1e-7
    eps_rel = 1e-3
    epsilon = 1e-7

    fast_integrals.compute_cos(
        ptr(partial_integral_cos), ptr(x_min), ptr(x_max), 
        ptr(k_n_values), ptr(t_values), ptr(z_values), 
        len(n_values), len(t_values), len(z_values), 
        limit, eps_abs, eps_rel, 
        config.omega, epsilon
    )

    fast_integrals.compute_sin(
        ptr(partial_integral_sin), ptr(x_min), ptr(x_max), 
        ptr(k_n_values), ptr(t_values), ptr(z_values), 
        len(n_values), len(t_values), len(z_values), 
        limit, eps_abs, eps_rel, 
        config.omega, epsilon
    )

    # Initialize the result array
    resummed_integral_sin = np.empty((len(n_values), len(t_values), len(z_values)))
    resummed_integral_cos = np.empty((len(n_values), len(t_values), len(z_values)))

    # Vectorized cumulative sum along the 't' axis (axis=1) to recover the full integrals
    resummed_integral_sin[:, :, :] = np.cumsum(partial_integral_sin[:, :, :], axis=1)
    resummed_integral_cos[:, :, :] = np.cumsum(partial_integral_cos[:, :, :], axis=1)


    # Initialize the result array
    result = np.empty((len(n_values), len(t_values), len(z_values)))
    result = np.sin(config.omega * t_values[None,:,None]) * resummed_integral_cos \
        - np.cos(config.omega * t_values[None,:,None]) * resummed_integral_sin # sin(tau-t) = cos(t)sin(tau) - sin(t)cos(tau)

    return result


def generate_coeffs(config):
    '''Computes the coefficients $c_n(t,z)$ for each allowed value of n,t,z for the parameters in config.
    
    Parameters
    ----------
    config : TalbotConfig
        The class containing all the parameters of the simulation.
    
    Returns
    -------
    coeffs : np.ndarray of floats
        The values of the coefficients $c_n(t,z)$. The array is of shape (N_max, N_t, N_z).
    '''

    # We store the range of t, z in the arrays of dimension (N_t) and (N_z)
    t_values = np.linspace(config.z_T/config.c * config.initial_t_zT, config.z_T/config.c * config.final_t_zT, config.N_t)
    z_values = np.linspace(0, config.N_z*config.delta_z, config.N_z)

    # We store the range of n, kn and gn in arrays of length N_max
    n_values = np.linspace(0, config.N_max-1, config.N_max, dtype=int)
    g_n_values = g_n_rect_delta(n_values, config)
    k_n_values = 2 * np.pi * n_values


    # We perform the integrals
    integrals_array = perform_integrals(config)


    # Component 1: - k_n_values[n] * integrals_array[n,t,z]
    integrals_part = - k_n_values[:,np.newaxis,np.newaxis] * z_values[np.newaxis,np.newaxis,:] * integrals_array

    # Component 2: sin(omega * (t_values[i] - z_values[j])) * Heaviside(t_values[i] - z_values[j])
    heaviside_part = np.sin(config.omega * (t_values[np.newaxis,:,np.newaxis] - z_values[np.newaxis,np.newaxis,:])) * np.heaviside(t_values[np.newaxis,:,np.newaxis] - z_values[np.newaxis,np.newaxis,:], 0.5)

    # Combine the components
    coeffs = g_n_values[:,np.newaxis,np.newaxis] * (integrals_part + heaviside_part)

    return coeffs


def generate_amplitude_field(config):
    '''Computes the value of the solution $u(t,x,z)$ for each allowed value of t, x and z for the parameters in config.
    
    Parameters
    ----------
    config : TalbotConfig
        The class containing all the parameters of the simulation.
    
    Returns
    -------
    field : np.ndarray of floats
        The values of the field $u(t,x,z)$ at each t,x,z. The array is of shape (N_t, N_x, N_z).
    '''


    # We compute the coefficients $c_n (t,z)$
    coeffs = generate_coeffs(config)

    # We compute the wave numbers
    k_n = 2 * np.pi * np.arange(config.N_max)

    # Create the grid for the cosine terms
    x_grid = np.arange(config.N_x) * config.delta_x
    cos_values = np.cos(np.outer(k_n, x_grid))
    del x_grid, k_n

    field = np.zeros([config.N_t, config.N_x, config.N_z])

    # Iterate over chunks of n axis to avoid memory problems
    chunk_size = 2  # Choose a reasonable chunk size based on available memory
    for i in tqdm(range(0, coeffs.shape[0], chunk_size)):
        chunk_coeffs = coeffs[i:i+chunk_size]
        chunk_cos_values = cos_values[i:i+chunk_size]
        
        # Compute the chunk of the contribution to E from the cosine term. 
        # For this we expand dimensions of coeffs and cos_values to match the desired broadcast shape.
        # New shapes:       (N_max, N_t, 1, N_z)        (N_max, 1, N_x,1)
        field_update_chunk = chunk_coeffs[:, :, np.newaxis, :] * chunk_cos_values[:, np.newaxis, :, np.newaxis]
        
        # Sum over n (along axis 0) and add to the field
        field += np.sum(field_update_chunk, axis=0)
        del field_update_chunk
    del cos_values, coeffs, chunk_coeffs, chunk_cos_values

    return field


def resize_field(field, config):
    '''Extends the domain of the solution from $0 \leq x \leq d/2$ to $-d \leq x \leq d$ using its symmetry.
    
    Parameters
    ----------
    field : np.ndarray of floats
        The values of the field $u(t,x,z)$ at each t,x,z for $0 \leq x \leq d/2$. The array is of shape (N_t, N_x, N_z).
    config : TalbotConfig
        The class containing all the parameters of the simulation.
    
    Returns
    -------
    resized_field : np.ndarray of floats
        The values of the field $u(t,x,z)$ at each t,x,z with for $-d \leq x \leq d$. Its shape is (N_t, 4 N_x, N_z).
    '''
    # We allocate the new array
    resized_field = np.empty([config.N_t, 4 * config.N_x, config.N_z])

    # Define an auxiliary array
    reversed_order = np.array([n for n in range(config.N_x - 1, -1, -1)])

    resized_field[:, 0:config.N_x, :] = field[:,0:config.N_x,:]
    resized_field[:,config.N_x:2*config.N_x,:] = field[:,reversed_order, :]
    resized_field[:,2*config.N_x:3*config.N_x,:] = field[:,0:config.N_x,:]
    resized_field[:,3*config.N_x:4*config.N_x,:] = field[:,reversed_order, :]
    return resized_field