import os
from tqdm import tqdm

from pyTalbot.benchmarks_talbot import TalbotConfig
from pyTalbot.talbot_utils import generate_amplitude_field, resize_field
from pyTalbot.video_maker import create_video_from_images, plot_field
from datetime import datetime

config = TalbotConfig()
print(config)

field = generate_amplitude_field(config)
field = field**2
field = resize_field(field, config)


# Photo destination
my_path = os.path.dirname(os.path.abspath(__file__))
results_path = os.path.join(my_path, 'results')
# Create the results folder if it doesn't exist
if not os.path.isdir(results_path):
    os.makedirs(results_path)

folder_name = 'd_λ=' + str(config.d/config._lambda) + '_w_λ=' + str(config.w/config._lambda) + '_' + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
folder_path = os.path.join(results_path, folder_name)
# Create the folder if it doesn't exist
if not os.path.isdir(folder_path):
    os.makedirs(folder_path)

# Save the parameters of the simulation into a file
with open(os.path.join(folder_path, 'parameters.txt'), 'w') as file:
    file.write(str(config)) 


for t_i in tqdm(range(0, config.N_t)):
    plot_field(t_i, field, config, folder_path, save_field = False)

if config.make_video:
    # Create video
    output_name = 'Talbot_carpet_d_λ=' + str(1/config._lambda) + '_w_λ=' + str(config.w/config._lambda) + '.mp4'
    create_video_from_images(folder_path, output_name)