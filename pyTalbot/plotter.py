import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from numpy import savetxt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_field(field, config, folder_path, title, file_name, save_field = False, difference = False, cmap = 'turbo', log_norm = True, log_min = 5e-3):
    '''Plots the field at time t_i and saves the image in folder_path as a PNG.

    Parameters
    ----------
    field : np.ndarray of floats
        Field to be plotted. Must of size (N_t, 4 N_x, N_z).
    config : TalbotConfig
        Class storing the parameters of the simulation.
    folder_path : string
        Path where the image must be stored. The folder MUST exist.
    title : string
        Title of the plot
    file_name : string
        Name of the file.
    save_field : bool, optional
        Whether the image should be saved as a txt file alongside the PNG image.
    difference : bool, optional
        Whether the values of the image run from -A^2 to A^2 (True) or from 0 to A^2 (False).
    cmap : str, optional
        Colormap used for the images. See documentation of matplotlib. Some recommended options are turbo and gray.
    log_norm : bool, optional
        Whether a loarithmic color map is used for the images. See documentation of matplotlib.
    log_min : float, optional
        Minimum value of the intensity displayed in the color bar.

    Returns
    -------
    None
    '''
    cm = 1/2.54
    fig, ax = plt.subplots(figsize=(32*cm, 18*cm))
    ax.set_title(title, fontsize = 20, y = 1.05)

    # We use the symmetry of the solution to extend the domain to -d <= x <= d
    resized_field = resize_field(field) 

    # Plot the Field
    # Some nice options are gray and turbo 
    if log_norm and not difference:
        im = ax.imshow(resized_field, cmap = cmap, norm=LogNorm(vmin=((config.d/config.l)**2)*log_min, vmax=(config.d/config.l)**2), interpolation = 'none')
    else:
        im = ax.imshow(resized_field, cmap = cmap, vmin=0, vmax=(config.d/config.l)**2, interpolation = 'none')

    # Label the Y axis and set the ticks
    ax.set_ylabel('Grating', fontsize = 18)
    ax.set_ylim(0, 4 * config.N_x)
    ticks_x = [0, config.N_x-1, 2 * config.N_x - 1, 3 * config.N_x - 1, 4 * config.N_x - 1]
    labels_x = ['$-d$', '$-\\dfrac{d}{2}$', '$0$', '$\\dfrac{d}{2}$', '$d$']
    ax.set_yticks(ticks_x)
    ax.set_yticklabels(labels_x, fontsize = 16)

    # Label the X axis and set the ticks
    ax.set_xlabel('Propagation of light $\longrightarrow$', fontsize = 18)
    ax.set_xlim(0, config.N_z - 1)
    ticks_z = [0, config.N_z/4, config.N_z/2, 3 * config.N_z/4, config.N_z]
    labels_z = ['$0$', '$\\dfrac{1}{4} Z_T$', '$\\dfrac{1}{2} Z_T$', '$\\dfrac{3}{4} Z_T$', '$Z_T$']
    ax.set_xticks(ticks_z)
    ax.set_xticklabels(labels_z, fontsize = 16)

    # Add the colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    if difference:
        cbar = fig.colorbar(im, cax=cax, ticks=[-(config.d/config.l)**2, -(config.d/config.l)**2/2, 0., (config.d/config.l)**2/2, (config.d/config.l)**2], fraction = 0.0458 * config.N_z/(4 * config.N_x), pad = 0.04, shrink = 0.9)
        cbar.set_label(label = 'Intensity of the field', fontsize = 18)
        cbar.ax.set_yticklabels(['$-A^2$', '$-\\dfrac{A^2}{2}$', '$0$', '$\\dfrac{A^2}{2}$', '$A^2$'], fontsize = 16)
    elif log_norm:
        cbar = fig.colorbar(im, cax=cax, ticks=[log_min*(config.d/config.l)**2, 0.01*(config.d/config.l)**2, 0.1*(config.d/config.l)**2, (config.d/config.l)**2], fraction = 0.0458 * config.N_z/(4 * config.N_x), pad = 0.04, shrink = 0.9)
        cbar.set_label(label = 'Intensity of the field', fontsize = 18)
        cbar.ax.set_yticklabels(['$0$', '$0.01 \\cdot A^2$', '$0.1 \\cdot A^2$', '$A^2$'], fontsize = 16)
    else:
        cbar = fig.colorbar(im, cax=cax, ticks=[0, 1/4*(config.d/config.l)**2, 1/2*(config.d/config.l)**2, 3/4*(config.d/config.l)**2, (config.d/config.l)**2], fraction = 0.0458 * config.N_z/(4 * config.N_x), pad = 0.04, shrink = 0.9)
        cbar.set_label(label = 'Intensity of the field', fontsize = 18)
        cbar.ax.set_yticklabels(['$0$', '$\\frac{1}{4} A^2$', '$\\frac{1}{2} A^2$', '$\\frac{3}{4} A^2$', '$A^2$'], fontsize = 16)
    #cbar.set_label(label = 'Amplitude of the field', fontsize = 18)
    #cbar.ax.set_yticklabels(['$-A$', '$-\\dfrac{A}{2}$', '$0$', '$\\dfrac{A}{2}$', '$A$'], fontsize = 16) 

    plt.savefig(os.path.join(folder_path, file_name), bbox_inches = 'tight', dpi = 300)  
    plt.close()

    if save_field:
        # Save the field at time t_i to a txt file
        savetxt(os.path.join(folder_path, file_name), resized_field, delimiter=',')


def plot_field_pdf(field, config, folder_path, title, file_name, save_field = False, difference = False, cmap = 'turbo', log_norm = True, log_min = 5e-3):
    '''Plots the field at time t_i and saves the image in folder_path as a PDF.

    Parameters
    ----------
    field : np.ndarray of floats
        Field to be plotted. Must of size (N_t, 4 N_x, N_z).
    config : TalbotConfig
        Class storing the parameters of the simulation.
    folder_path : string
        Path where the image must be stored. The folder MUST exist.
    title : string
        Title of the plot
    file_name : string
        Name of the file.
    save_field : bool, optional
        Whether the image should be saved as a txt file alongside the PDF image.
    difference : bool, optional
        Whether the values of the image run from -A^2 to A^2 (True) or from 0 to A^2 (False).
    cmap : str, optional
        Colormap used for the images. See documentation of matplotlib. Some recommended options are turbo and gray.
    log_norm : bool, optional
        Whether a loarithmic color map is used for the images. See documentation of matplotlib.
    log_min : float, optional
        Minimum value of the intensity displayed in the color bar.

    Returns
    -------
    None
    '''
    cm = 1/2.54
    fig, ax = plt.subplots(figsize=(32*cm, 18*cm))

    # We use the symmetry of the solution to extend the domain to -d <= x <= d
    resized_field = resize_field(field) 

    # Plot the Field
    # Some nice options are gray and turbo 
    if log_norm and not difference:
        im = ax.imshow(resized_field, cmap = cmap, norm=LogNorm(vmin=((config.d/config.l)**2)*log_min, vmax=(config.d/config.l)**2), interpolation = 'none')
    else:
        im = ax.imshow(resized_field, cmap = cmap, vmin=0, vmax=(config.d/config.l)**2, interpolation = 'none')

    # Label the X axis and set the ticks
    ax.set_ylabel('Grating', fontsize = 18)
    ax.set_ylim(0, 4 * config.N_x)
    ticks_x = [0, config.N_x-1, 2 * config.N_x - 1, 3 * config.N_x - 1, 4 * config.N_x - 1]
    labels_x = ['$-d$', '$-\\dfrac{d}{2}$', '$0$', '$\\dfrac{d}{2}$', '$d$']
    ax.set_yticks(ticks_x)
    ax.set_yticklabels(labels_x, fontsize = 16)

    # Label the Y axis and set the ticks
    ax.set_xlabel('Propagation of light $\longrightarrow$', fontsize = 18)
    ax.set_xlim(0, config.N_z - 1)
    ticks_z = [0, config.N_z/4, config.N_z/2, 3 * config.N_z/4, config.N_z]
    labels_z = ['$0$', '$\\dfrac{1}{4} Z_T$', '$\\dfrac{1}{2} Z_T$', '$\\dfrac{3}{4} Z_T$', '$Z_T$']
    ax.set_xticks(ticks_z)
    ax.set_xticklabels(labels_z, fontsize = 16)

    # Add the colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    if difference:
        cbar = fig.colorbar(im, cax=cax, ticks=[-(config.d/config.l)**2, -(config.d/config.l)**2/2, 0., (config.d/config.l)**2/2, (config.d/config.l)**2], fraction = 0.0458 * config.N_z/(4 * config.N_x), pad = 0.04, shrink = 0.9)
        cbar.set_label(label = 'Intensity of the field', fontsize = 18)
        cbar.ax.set_yticklabels(['$-A^2$', '$-\\dfrac{A^2}{2}$', '$0$', '$\\dfrac{A^2}{2}$', '$A^2$'], fontsize = 16)
    elif log_norm:
        cbar = fig.colorbar(im, cax=cax, ticks=[log_min*(config.d/config.l)**2, 0.01*(config.d/config.l)**2, 0.1*(config.d/config.l)**2, (config.d/config.l)**2], fraction = 0.0458 * config.N_z/(4 * config.N_x), pad = 0.04, shrink = 0.9)
        cbar.set_label(label = 'Intensity of the field', fontsize = 18)
        cbar.ax.set_yticklabels(['$0$', '$0.01 \\cdot A^2$', '$0.1 \\cdot A^2$', '$A^2$'], fontsize = 16)
    else:
        cbar = fig.colorbar(im, cax=cax, ticks=[0, 1/4*(config.d/config.l)**2, 1/2*(config.d/config.l)**2, 3/4*(config.d/config.l)**2, (config.d/config.l)**2], fraction = 0.0458 * config.N_z/(4 * config.N_x), pad = 0.04, shrink = 0.9)
        cbar.set_label(label = 'Intensity of the field', fontsize = 18)
        cbar.ax.set_yticklabels(['$0$', '$\\frac{1}{4} A^2$', '$\\frac{1}{2} A^2$', '$\\frac{3}{4} A^2$', '$A^2$'], fontsize = 16)
    #cbar.set_label(label = 'Amplitude of the field', fontsize = 18)
    #cbar.ax.set_yticklabels(['$-A$', '$-\\dfrac{A}{2}$', '$0$', '$\\dfrac{A}{2}$', '$A$'], fontsize = 16) 

    plt.savefig(os.path.join(folder_path, file_name), bbox_inches = 'tight', dpi = 300)  
    plt.close()

    if save_field:
        # Save the field at time t_i to a txt file
        savetxt(os.path.join(folder_path, file_name), resized_field, delimiter=',')



def video_from_images(images_path, output_name, fps=24):
    '''Creates the video showcasing the formation of the Talbot effect.

    Parameters
    ----------
    images_path : string
        Path where the images are stored. The images name must be sorted and end with .png to be identified by this function. The video will be stored in this same path.
    output_name : string
        Name of the output file.
    fps : int, optional
        Framerate of the video

    Returns
    -------
    None
    '''

    if not os.path.exists(os.path.dirname(images_path)): # We make sure that the images_path exists.
        os.makedirs(os.path.dirname(images_path))
        
    command = f'ffmpeg -framerate {fps} -pattern_type glob -i "{images_path}/*.png" -vf "scale=3288:1928" -c:v libx264 -pix_fmt yuv420p "{output_name}"'
    os.system(command) # We create the video



def resize_field(field):
    '''Extends the domain of the solution from $0 \leq x \leq d/2$ to $-d \leq x \leq d$ using its symmetry.
    
    Parameters
    ----------
    field : np.ndarray of floats
        The values of the field $u(t,x,z)$ at each x and z for $0 \leq x \leq d/2$. The array is of shape (N_x, N_z).
    
    Returns
    -------
    resized_field : np.ndarray of floats
        The values of the field $u(t,x,z)$ at each x and z with for $-d \leq x \leq d$. Its shape is (4 N_x, N_z).
    '''
    # We allocate the new array
    N_x = field.shape[0]
    resized_field = np.empty((4 * N_x, field.shape[1]), dtype=field.dtype)

    resized_field[0:N_x, :] = field
    resized_field[2*N_x:3*N_x,:] = field

    resized_field[N_x:2*N_x, :] = field[::-1, :]
    resized_field[3*N_x:4*N_x, :] = field[::-1, :]

    return resized_field