import os, shutil
from tqdm import tqdm

from pyTalbot.talbotconfig import TalbotConfig
from pyTalbot.transient_utils import generate_amplitude_field
from pyTalbot.stationary_utils import generate_stationary_amplitude_field
from pyTalbot.plotter import video_from_images, plot_field
from datetime import datetime
from mpi4py import MPI

# We kill the programme if it's not the master
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


##################################
# Configuration of the simulation.
##################################

make_stationary = True # Do we plot the final field considering the stationary approximation?
make_transient = True # Do we want to calculate the transient behaviour? 
make_video = True # Do we make a video? NEEDS FFMPEG installed
clear_cache = False # Do we clear the images at the end? CAREFULL, THIS DELETES ALL FILES IN THE CACHE FOLDER
colour = 'turbo' # Colour of the plots. Must be str. Suggested picks are 'gray' and 'turbo'. Check the documentation of matplotlib for more.


# We print the parameters of the simulation
config = TalbotConfig()

if rank == 0:
    print(config)


    # Where define the location of the results folder
    my_path = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(my_path, 'results')
    if not os.path.isdir(results_path): # Create the results folder if it doesn't exist
        os.makedirs(results_path)

    # Where create the folder to store the simulation's output
    folder_name = 'd_lambda=' + str(config.d/config._lambda) + '_w_lambda=' + str(config.w/config._lambda) + '_' + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    folder_path = os.path.join(results_path, folder_name)
    if not os.path.isdir(folder_path): # Create the folder if it doesn't exist
        os.makedirs(folder_path)

    # Save the parameters of the simulation into a file
    with open(os.path.join(folder_path, 'parameters.txt'), 'w') as file:
        file.write(str(config)) 


if make_transient:
    # We compute the intensity of the light and extend it to -d <= x <= d
    field = generate_amplitude_field(config) # We compute the amplitude of the solution
    field = field**2 # We compute the intensity of the light
    
    # We create the caché folder if it doesn't exist
    cache_path = os.path.join(folder_path, 'cache')
    if not os.path.isdir(cache_path):
        os.makedirs(cache_path)

    # We plot the solution at each time t_i at cache
    for t_i in tqdm(range(0, config.N_t)):
        title = 'Intensity of the field at $t = ' + str(round(t_i * config.delta_t/(config.z_T),4)) + '\\, Z_T/c$ for $\\frac{d}{\\lambda}='+str(1/config._lambda)+'$ and $\\frac{w}{\\lambda}=' + str(config.w/config._lambda)+'$'
        file_name = 'd_lambda=' + str(1/config._lambda) + '_w_lambda=' + str(config.w/config._lambda)+'_' + str(t_i).rjust(len(str(config.N_t)),'0') + '_carpet.png'
        plot_field(field[t_i], config, cache_path, title, file_name, save_field = False, cmap = colour)

    # We plot the final image also somewhere else to store it
    final_field = field[config.N_t - 1]
    del field
    file_name = 'd_lambda=' + str(1/config._lambda) + '_w_lambda=' + str(config.w/config._lambda)+'_TRANSIENT_carpet.png'
    plot_field(final_field, config, folder_path, title, file_name, save_field = False, cmap = colour)

    # We make the video
    if make_video:
        output_name = 'Talbot_carpet_d_lambda=' + str(1/config._lambda) + '_w_lambda=' + str(config.w/config._lambda) + '.mp4'
        output_path = os.path.join(folder_path, output_name)
        video_from_images(cache_path, output_path)

    # We clear the caché
    if clear_cache:
        shutil.rmtree(cache_path)


# We make the stationary image
if make_stationary:
    stationary_field = generate_stationary_amplitude_field(config)
    stationary_field = stationary_field**2 # We compute the intensity of the light

    title = 'Intensity of the stationary field at $t = ' + str(round((config.N_t-1) * config.delta_t/(config.z_T),4)) + '\\, Z_T/c$ for $\\frac{d}{\\lambda}='+str(1/config._lambda)+'$ and $\\frac{w}{\\lambda}=' + str(config.w/config._lambda)+'$'
    file_name = 'd_lambda=' + str(1/config._lambda) + '_w_lambda=' + str(config.w/config._lambda)+'_STATIONARY_carpet.png'
    plot_field(stationary_field, config, folder_path, title, file_name, save_field = False, cmap = colour) # We plot the solution

    if make_transient:
        field_difference = stationary_field - final_field
        title = 'Difference of the intensity of the stationary and transient fields at $t = ' + str(round((config.N_t-1) * config.delta_t/(config.z_T),4)) + '\\, Z_T/c$ for $\\frac{d}{\\lambda}='+str(1/config._lambda)+'$ and $\\frac{w}{\\lambda}=' + str(config.w/config._lambda)+'$'
        file_name = 'd_lambda=' + str(1/config._lambda) + '_w_lambda=' + str(config.w/config._lambda)+'_DIFFERENCE_carpet.png'
        plot_field(field_difference, config, folder_path, title, file_name, save_field = False, difference = True, cmap = colour) # We plot the difference between the stationary and transient case