import numpy as np



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
    result = np.where(n == 0, 1, 2*np.sin(n * np.pi * config.l) / (np.pi * n * config.l)) # We multiply by 2 to account at the same time for g_n and g_(-n)
    np.seterr(**original_settings)
    return result


def stationary_coeff(k_n,t,z,omega):
    '''Computes the coefficients $c_n(t,z)$ for each allowed value of n,z for the parameters in config.
    
    Parameters
    ----------
    k_n : np.ndarray of floats
        The values of the k_n.
    t : float
        The time at which we compute the field.
    z : np.ndarray of floats
        The zs at which we compute the field.
    omega : float
        The angular frequency of the source
    
    Returns
    -------
    coeff : np.ndarray of floats
        The values of the coefficients $c_n(t,z)$ without the g_n factor according to the stationary approximation in equation (5.4).
    '''
    s = np.sqrt(np.abs(omega**2 - k_n**2)) # We precompute this to avoid sqrt(-x)

    return np.where(k_n < omega, 
                      np.sin(omega * t - z * s),  # for k_n < omega
                      np.sin(omega * t) * np.exp( - z * s)  # for k_n >= omega
                     )



def generate_stationary_coeffs(config):
    '''Computes the coefficients $c_n(t,z)$ for each allowed value of n,z for the parameters in config.
    
    Parameters
    ----------
    config : TalbotConfig
        The class containing all the parameters of the simulation.
    
    Returns
    -------
    coeffs : np.ndarray of floats
        The values of the coefficients $c_n(t,z)$. The array is of shape (N_max, N_z).
    '''

    # We store the range of t, z in the arrays of dimension (N_t) and (N_z)
    t = config.z_T/config.c * config.final_t_zT
    z_values = np.linspace(0, config.N_z*config.delta_z, config.N_z)

    # We store the range of n, kn and gn in arrays of length N_max
    n_values = np.linspace(0, config.N_max-1, config.N_max, dtype=int)
    g_n_values = g_n_rect_delta(n_values, config)
    k_n_values = 2 * np.pi * n_values


    # Combine the components
    coeffs = g_n_values[:,np.newaxis] * stationary_coeff(k_n_values[:,np.newaxis], t, z_values[np.newaxis, :], config.omega)

    return coeffs


def generate_stationary_amplitude_field(config):
    '''Computes the value of the solution $u(t,x,z)$ for each allowed value of x and z for the parameters in config.
    
    Parameters
    ----------
    config : TalbotConfig
        The class containing all the parameters of the simulation.
    
    Returns
    -------
    field : np.ndarray of floats
        The values of the field $u(t,x,z)$ at each x,z. The array is of shape (N_x, N_z).
    '''


    # We compute the coefficients $c_n (t,z)$
    coeffs = generate_stationary_coeffs(config)

    # We compute the wave numbers
    k_n = 2 * np.pi * np.arange(config.N_max)

    # Create the grid for the cosine terms
    x_grid = np.arange(config.N_x) * config.delta_x
    cos_values = np.cos(np.outer(k_n, x_grid))
    del x_grid, k_n

    field = np.zeros([config.N_x, config.N_z])

    # Compute the chunk of the contribution to E from the cosine term. 
    # For this we expand dimensions of coeffs and cos_values to match the desired broadcast shape.
    # New shapes:       (N_max, 1, N_z)        (N_max, N_x,1)
    field_update = coeffs[:, np.newaxis, :] * cos_values[:, :, np.newaxis]
    del cos_values, coeffs
    
    # Sum over n (along axis 0) and add to the field
    field += np.sum(field_update, axis=0)


    return field


def stationary_energy_coeff(k_n,t,z,omega):
    '''Computes the coefficients $c_n(t,z)$ for each allowed value of n,z for the parameters in config.
    
    Parameters
    ----------
    k_n : np.ndarray of floats
        The values of the k_n.
    t : float
        The time at which we compute the field.
    z : np.ndarray of floats
        The zs at which we compute the field.
    omega : float
        The angular frequency of the source
    
    Returns
    -------
    coeff : np.ndarray of floats
        The values of the coefficients $c_n(t,z)$ without the g_n factor according to the stationary approximation in equation (5.4).
    '''
    s = np.sqrt(np.abs(omega**2 - k_n**2)) # We precompute this to avoid sqrt(-x)

    return np.where(k_n < omega, 
                      np.exp(1j * (omega * t - z * s)),  # for k_n < omega
                      np.exp(1j * omega * t) * np.exp( - z * s)  # for k_n >= omega
                     )
                


def generate_stationary_energy_coeffs(config):
    '''Computes the coefficients $c_n(t,z)$ for each allowed value of n,z for the parameters in config.
    
    Parameters
    ----------
    config : TalbotConfig
        The class containing all the parameters of the simulation.
    
    Returns
    -------
    coeffs : np.ndarray of floats
        The values of the coefficients $c_n(t,z)$. The array is of shape (N_max, N_z).
    '''

    # We store the range of t, z in the arrays of dimension (N_t) and (N_z)
    t = config.z_T/config.c * config.final_t_zT
    z_values = np.linspace(0, config.N_z*config.delta_z, config.N_z)

    # We store the range of n, kn and gn in arrays of length N_max
    n_values = np.linspace(0, config.N_max-1, config.N_max, dtype=int)
    g_n_values = g_n_rect_delta(n_values, config)
    k_n_values = 2 * np.pi * n_values


    # Combine the components
    coeffs = g_n_values[:,np.newaxis] * stationary_energy_coeff(k_n_values[:,np.newaxis], t, z_values[np.newaxis, :], config.omega)

    return coeffs


def generate_stationary_energy_amplitude_field(config):
    '''Computes the value of the solution $u(t,x,z)$ for each allowed value of x and z for the parameters in config.
    
    Parameters
    ----------
    config : TalbotConfig
        The class containing all the parameters of the simulation.
    
    Returns
    -------
    field : np.ndarray of floats
        The values of the field $u(t,x,z)$ at each x,z. The array is of shape (N_x, N_z).
    '''


    # We compute the coefficients $c_n (t,z)$
    coeffs = generate_stationary_energy_coeffs(config)

    # We compute the wave numbers
    k_n = 2 * np.pi * np.arange(config.N_max)

    # Create the grid for the cosine terms
    x_grid = np.arange(config.N_x) * config.delta_x
    cos_values = np.cos(np.outer(k_n, x_grid))
    del x_grid, k_n

    field = np.zeros([config.N_x, config.N_z])
    field = field.astype(np.complex128)

    # Compute the chunk of the contribution to E from the cosine term. 
    # For this we expand dimensions of coeffs and cos_values to match the desired broadcast shape.
    # New shapes:       (N_max, 1, N_z)        (N_max, N_x,1)
    field_update = coeffs[:, np.newaxis, :] * cos_values[:, :, np.newaxis]
    del cos_values, coeffs
    
    # Sum over n (along axis 0) and add to the field
    field += np.sum(field_update, axis=0)


    return field