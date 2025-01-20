import os
from tqdm import tqdm

from pyTalbot.benchmarks_talbot import TalbotConfig
from pyTalbot.talbot_utils import generate_amplitude_field, resize_field
from pyTalbot.video_maker import create_video_from_images, plot_field
from datetime import datetime


# Configuration of the simulation.
make_video = True # Do we make a video? NEEDS FFMPEG installed
clear_images = False # Do we clear the images at the end? CAREFULL, THIS DELETES ALL .PNG FILES IN THE DESTINATION FOLDER


# We print the parameters of the simulation
config = TalbotConfig()
print(config)

# We compute the intensity of the light and extend it to -d <= x <= d
field = generate_amplitude_field(config) # We compute the amplitude of the solution
field = field**2 # We compute the intensity of the light
field = resize_field(field, config) # We use the symmetry of the solution to extend the domain to -d <= x <= d


# Where define the location of the results folder
my_path = os.path.dirname(os.path.abspath(__file__))
results_path = os.path.join(my_path, 'results')
if not os.path.isdir(results_path): # Create the results folder if it doesn't exist
    os.makedirs(results_path)

# Where create the folder to store the simulation's output
folder_name = 'd_λ=' + str(config.d/config._lambda) + '_w_λ=' + str(config.w/config._lambda) + '_' + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
folder_path = os.path.join(results_path, folder_name)
if not os.path.isdir(folder_path): # Create the folder if it doesn't exist
    os.makedirs(folder_path)

# Save the parameters of the simulation into a file
with open(os.path.join(folder_path, 'parameters.txt'), 'w') as file:
    file.write(str(config)) 


# We plot the solution at each time t_i
for t_i in tqdm(range(0, config.N_t)):
    plot_field(t_i, field, config, folder_path, save_field = False)

# We make the video
if make_video:
    output_name = 'Talbot_carpet_d_λ=' + str(1/config._lambda) + '_w_λ=' + str(config.w/config._lambda) + '.mp4'
    create_video_from_images(folder_path, output_name)

# We clear the images in folder_path
if clear_images:
    command = f'find "{folder_path}" -type f -iname \*.png -delete'
    os.system(command) # We delete all .png files in folder_path