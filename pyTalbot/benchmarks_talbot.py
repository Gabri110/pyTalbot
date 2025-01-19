import io
import numpy as np


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
        self.N_t = 250 # Number of samples in time
        self.N_max = int(self.d / self._lambda * 5) # Number of terms in the series

        # Other relevant magnitudes
        self.initial_t_zT = 0. # Initial time / Z_t
        self.final_t_zT = 2. # Final time / Z_t
        self.delta_t = self.z_T/self.c/(self.N_t-1) * (self.final_t_zT - self.initial_t_zT) # Time between photos
        self.delta_x = self.d/2/self.N_x # X-Distance between points
        self.delta_z = self.z_T/self.N_z # Z-Distance between points

        self.make_video = True # Do we make a video?


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
            "Initial time / z_T": self.initial_t_zT,
            "Final time / z_T": self.final_t_zT
        }
        
        # Create a string stream to capture the print output
        output = io.StringIO()
        
        # Print to the string stream instead of the console
        output.write("{:<45} {:<40}\n".format('Parameter', 'Value'))
        output.write("-" * 65 + "\n")
        
        for key, value in params.items():
            output.write("{:<45} {:<40}\n".format(key, value))
        
        # Get the string from the output stream
        result = output.getvalue()
        
        # Close the string stream
        output.close()
        
        return result