import os
import numpy as np
from tqdm import tqdm
import numpy as np

from talbot_utils import generate_amplitude_field, resize_field, plot_field
from video_maker import pdf_to_images, create_video_from_images
from datetime import datetime


class TalbotConfig:
    def __init__(self):
        self.A = 1. # Amplitude of signal
        self.c = 1. # Speed of light
        self.d = 1. # Distance between gratings we fix it = 1
        self._lambda = self.d / 10. # Wavelength
        self.w = 2 * self._lambda # Width of the gratings

        # Other relevant magnitudes
        self.omega = 2 * np.pi * self.c / self._lambda # Frequency of the signal
        self.z_T = self._lambda/(1. - np.sqrt(1.-(self._lambda/self.d) ** 2)) # Talbot distance = 2 d^2/λ

        # Simulation parameters
        self.N_x = 27*10 # Number of samples in x direction
        self.N_z = 192*10 # Number of samples in z direction
        self.N_t = 500 # Number of samples in time
        self.N_max = int(self.d / self._lambda * 4) # Number of terms in the series

        # Other relevant magnitudes
        self.last_t_zT = 1. # Final time / Z_t
        self.delta_t = self.z_T/self.c/(self.N_t-1) * self.last_t_zT # Time between photos
        self.delta_x = self.d/2/self.N_x # X-Distance between points
        self.delta_z = self.z_T/self.N_z # Z-Distance between points


    def __str__(self):
        params = {
            "Amplitude of signal (A)": self.A,
            "Speed of light (c)": self.c,
            "Distance between gratings (d)": self.d,
            "Wavelength (lambda)": self._lambda,
            "Width of the gratings (w)": self.w,
            "Frequency of the signal (omega)": self.omega,
            "Talbot distance (z_T)": self.z_T,
            "Number of samples in x direction (N_x)": self.N_x,
            "Number of samples in z direction (N_z)": self.N_z,
            "Number of samples in time (N_t)": self.N_t,
            "Number of terms in the series (N_max)": self.N_max,
            "Time between photos (delta_t)": self.delta_t,
            "X-Distance between points (delta_x)": self.delta_x,
            "Z-Distance between points (delta_z)": self.delta_z,
            "Final time / z_T": self.last_t_zT
        }
        
        print("{:<45} {:<40}".format('\nParameter', 'Value'))
        print("-" * 65)
        for key, value in params.items():
            print("{:<45} {:<40}".format(key, value))
        return ""
    
    def debugging(self):
        if self.Debugging:
            # Simulation parameters
            self.N_x = 5 # Number of samples in x direction
            self.N_z = 5 # Number of samples in z direction
            self.N_t = 5 # Number of samples in time
            self.N_max = int(self.d / self._lambda)*2 # Number of terms in the series

        return

if __name__ == "__main__":
    config = TalbotConfig()

    # Are we debugging?
    config.Debugging = True
    config.debugging()

    print(config)

    # Photo destination
    my_path = os.path.dirname(os.path.abspath(__file__))
    folder_name = 'd_λ=' + str(config.d/config._lambda) + '_w_λ=' + str(config.w/config._lambda) + str(datetime.now())
    folder_path = os.path.join(my_path, folder_name)

    # Create the folder if it doesn't exist
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    field = generate_amplitude_field(config)
    field = field**2
    field = resize_field(field, config)

    for t_i in tqdm(range(0, config.N_t)):
        plot_field(t_i, field, config, folder_path, save_field = False)

    # Convert PDFs to images
    image_files = pdf_to_images(folder_path)

    # Create video
    output_path = os.path.join(folder_path, 'Talbot_carpet_d_λ=' + str(1/config._lambda) + '_w_λ=' + str(config.w/config._lambda)+'_' + str(t_i) + '.mp4')
    create_video_from_images(image_files, output_path)
