import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1
from scipy.integrate import quad
from scipy import LowLevelCallable
from numba import njit,cfunc, carray # Remember to also have numba-scipy installed!!!!!
from numba.types import intc, CPointer, float64

from tqdm import tqdm


def jit_integrand_function(integrand_function):
    jitted_function = njit(integrand_function,error_model="numpy",fastmath=True)

    @cfunc(float64(intc, CPointer(float64)))
    def wrapped(n, xx):
        ar = carray(xx, n)
        
        return jitted_function(ar[0], ar[1:])
    return LowLevelCallable(wrapped.ctypes)



def perform_integrals(config):
    '''
    We perform the integrals using quad
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


    # We define the integrand and JIT it with Numba (and numba-scipy) for a faster performance
    @jit_integrand_function
    def integrand_sin(tau, args):
        k,z,omega, epsilon = args 
        u = np.sqrt(np.maximum(0,tau**2 - z**2))
        if u < epsilon:
            return np.sin(omega * tau)*k/2
        else:
            return np.sin(omega * tau) * j1(k * u) / u # TO UPDATE
        
    @jit_integrand_function
    def integrand_cos(tau, args):
        k,z,omega, epsilon = args 
        u = np.sqrt(np.maximum(0,tau**2 - z**2))
        if u < epsilon:
            return np.cos(omega * tau)*k/2
        else:
            return np.cos(omega * tau) * j1(k * u) / u # TO UPDATE
        


    # NON-VECTORISED
    partial_integral_sin = np.zeros((len(n_values),len(t_values),len(z_values)))
    partial_integral_cos = np.zeros((len(n_values),len(t_values),len(z_values)))

    for n in tqdm(range(1, len(n_values))): # We skip the n=0 case as it is trivially = 0
        for i in range(1, len(t_values)): # We skip the t=0 case as it is trivially = 0
            for j in range(1, len(z_values)): # We skip the z=0 case as it is trivially = 0
                partial_integral_sin[n,i,j],_ = quad(integrand_sin, x_min[i,j], x_max[i,j], args=(k_n_values[n],z_values[j],config.omega, 1e-7), limit=10000, epsabs=1e-7, epsrel=1e-4) # We use quad
                partial_integral_cos[n,i,j],_ = quad(integrand_cos, x_min[i,j], x_max[i,j], args=(k_n_values[n],z_values[j],config.omega, 1e-7), limit=10000, epsabs=1e-7, epsrel=1e-4) # We use quad


    # Initialize the result array
    resummed_integral_sin = np.zeros((len(n_values), len(t_values), len(z_values)))
    resummed_integral_cos = np.zeros((len(n_values), len(t_values), len(z_values)))

    # Vectorized cumulative sum along the 't' axis (axis=1)
    resummed_integral_sin[:, :, :] = np.cumsum(partial_integral_sin[:, :, :], axis=1)
    resummed_integral_cos[:, :, :] = np.cumsum(partial_integral_cos[:, :, :], axis=1)


    # Initialize the result array
    result = np.empty((len(n_values), len(t_values), len(z_values)))
    result = np.sin(config.omega * t_values[None,:,None]) * resummed_integral_cos - np.cos(config.omega * t_values[None,:,None]) * resummed_integral_sin

    return result



def g_n_rect_delta(n, config):
    # Rect function NORMALISED so that it tends to the delta function 
    original_settings = np.seterr()
    np.seterr(divide='ignore', invalid='ignore')
    result = np.where(n == 0, 1, 2*np.sin(n * np.pi * config.w) / (np.pi * n * config.w**2)) # We multiply by 2 to account for the +- 
    np.seterr(**original_settings)
    return result


def generate_coeffs(config):
    '''
    This function computes the coefficients c_n(t,z) for each allowed value of n,t,z for the parameters in config.
    We implicitly use the pylevin package and the integrate routine to take care of the integration.
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
    coeffs = generate_coeffs(config)

    k_n = 2 * np.pi * np.arange(config.N_max)

    # Create the grid for the cosine terms
    x_grid = np.arange(config.N_x) * config.delta_x
    cos_values = np.cos(np.outer(k_n, x_grid))
    del x_grid, k_n

    field = np.zeros([config.N_t, config.N_x, config.N_z])
    
    #field_update = coeffs[:, :, np.newaxis, :] * cos_values[:, np.newaxis, :, np.newaxis]


    # Iterate over chunks of n axis to avoid memory problems
    chunk_size = 3  # Choose a reasonable chunk size based on available memory
    for i in tqdm(range(0, coeffs.shape[0], chunk_size)):
        chunk_coeffs = coeffs[i:i+chunk_size]
        chunk_cos_values = cos_values[i:i+chunk_size]
        
        # Compute the chunk of the contribution to E from the cosine term. 
        # For this we expand dimensions of coeffs and cos_values to match the desired broadcast shape.
        # New shapes:       (N_max, N_t, 1, N_z)        (N_max, 1, N_x,1)
        field_update_chunk = chunk_coeffs[:, :, np.newaxis, :] * chunk_cos_values[:, np.newaxis, :, np.newaxis]
        
        # Sum over n (along axis 0) and add to the field
        field += np.sum(field_update_chunk, axis=0)
    del cos_values, coeffs, chunk_coeffs, chunk_cos_values, field_update_chunk

    # Sum over n (along axis 0) to update E
    #field = np.sum(field_update, axis = 0)  # Shape: (N_t, N_x, N_z)
    return field


def resize_field(field, config):
    resized_field = np.empty([config.N_t, 4 * config.N_x, config.N_z])
    reversed_order = np.array([n for n in range(config.N_x - 1, -1, -1)])
    resized_field[:, 0:config.N_x, :] = field[:,0:config.N_x,:]
    resized_field[:,config.N_x:2*config.N_x,:] = field[:,reversed_order, :]
    resized_field[:,2*config.N_x:3*config.N_x,:] = field[:,0:config.N_x,:]
    resized_field[:,3*config.N_x:4*config.N_x,:] = field[:,reversed_order, :]
    return resized_field


def plot_field(t_i, field, config, folder_path, save_field = False):
    cm = 1/2.54
    plt.figure(figsize=(32*cm, 18*cm))
    plt.title('Gauge Field at $t = ' + str(round(t_i * config.delta_t/(config.z_T),4)) + '\\, Z_T/c$ for $\\frac{d}{\\lambda}='+str(1/config._lambda)+'$ and $\\frac{w}{\\lambda}=' + str(config.w/config._lambda)+'$', fontsize = 20, y = 1.05)
    
    # Plot the Gauge Field
    im = plt.imshow(field[t_i], cmap = 'gray', vmin = 0, vmax = (config.d/config.w)**2, interpolation = 'none')

    # Label the X axis and set the ticks
    plt.ylabel('Grating', fontsize = 18)
    plt.ylim(0, 4 * config.N_x)
    ticks_x = [0, config.N_x-1, 2 * config.N_x - 1, 3 * config.N_x - 1, 4 * config.N_x - 1]
    labels_x = ['$-d$', '$-\\dfrac{d}{2}$', '$0$', '$\\dfrac{d}{2}$', '$d$']
    plt.yticks(ticks_x, labels_x, fontsize = 16)
    
    # Label the Y axis and set the ticks
    plt.xlabel('Propagation of light ---->', fontsize = 18)
    plt.xlim(0, config.N_z - 1)
    ticks_z = [0, config.N_z/4, config.N_z/2, 3 * config.N_z/4, config.N_z]
    labels_z = ['$0$', '$\\dfrac{1}{4} Z_T$', '$\\dfrac{1}{2} Z_T$', '$\\dfrac{3}{4} Z_T$', '$Z_T$']
    plt.xticks(ticks_z, labels_z, fontsize = 16)
    
    # Add the colorbar
    cbar = plt.colorbar(im, ticks=[0., (config.d/config.w)**2/4, (config.d/config.w)**2/2, 3*(config.d/config.w)**2/4, (config.d/config.w)**2], fraction = 0.0458 * config.N_z/(4 * config.N_x), pad = 0.04, shrink = 0.9)
    cbar.set_label(label = 'Intensity of the field', fontsize = 18)
    cbar.ax.set_yticklabels(['$0$', '$\\dfrac{A^2}{4}$', '$\\dfrac{A^2}{2}$', '$\\dfrac{3A^2}{4}$', '$A^2$'], fontsize = 16)
    #cbar.set_label(label = 'Amplitude of the field', fontsize = 18)
    #cbar.ax.set_yticklabels(['$-A$', '$-\\dfrac{A}{2}$', '$0$', '$\\dfrac{A}{2}$', '$A$'], fontsize = 16)

    file_name = 'd_λ=' + str(1/config._lambda) + '_w_λ=' + str(config.w/config._lambda)+'_' + str(t_i) + '_carpet.png'
    plt.savefig(os.path.join(folder_path, file_name), bbox_inches = 'tight', dpi = 300)  
    plt.close()

    if save_field:
        # Save the field at time t_i to a txt file
        txt_file_name = 'd_λ=' + str(1/config._lambda) + '_w_λ=' + str(config.w/config._lambda)+'_' + str(t_i) + '_carpet.txt'
        np.savetxt(os.path.join(folder_path, txt_file_name), field[t_i], delimiter=',')