import numpy as np
from tqdm import tqdm
from pyTalbot.perform_integrals import perform_integrals


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
    result = np.where(n == 0, 2, 2 * 2 * np.sin(n * np.pi * config.w) / (np.pi * n * config.w)) # We multiply by 2 to account at the same time for g_n and g_(-n)
    np.seterr(**original_settings)
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


    field = np.empty([config.N_t, config.N_x, config.N_z])

    # Iterate over chunks of z axis to avoid memory problems
    chunk_size = 4  # Choose a reasonable chunk size based on available memory
    for z in tqdm(range(0, config.N_z, chunk_size)):
        z_end = min(z + chunk_size, config.N_z)
        chunk_coeffs = coeffs[:, :, z:z_end]
        
        # Compute the chunk of the contribution to the field from the cosine term. 
        # New shapes:       (N_max, N_t, chunk_size)        (N_max, N_x, 1)
        field_update_chunk = chunk_coeffs[:, :, np.newaxis, :] * cos_values[:, np.newaxis, :, np.newaxis]
        
        # Sum over n (along axis 0) and add to the field
        field[:, :, z:z_end] = np.sum(field_update_chunk, axis=0)
        del field_update_chunk

    del cos_values, coeffs, chunk_coeffs

    return field