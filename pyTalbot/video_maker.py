import os
import matplotlib.pyplot as plt
from numpy import savetxt


def plot_field(t_i, field, config, folder_path, save_field = False):
    cm = 1/2.54
    plt.figure(figsize=(32*cm, 18*cm))
    plt.title('Intensity of the Field at $t = ' + str(round(t_i * config.delta_t/(config.z_T),4)) + '\\, Z_T/c$ for $\\frac{d}{\\lambda}='+str(1/config._lambda)+'$ and $\\frac{w}{\\lambda}=' + str(config.w/config._lambda)+'$', fontsize = 20, y = 1.05)
    
    # Plot the Field
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

    file_name = 'd_λ=' + str(1/config._lambda) + '_w_λ=' + str(config.w/config._lambda)+'_' + str(t_i).rjust(len(str(config.N_t)),'0') + '_carpet.png'
    plt.savefig(os.path.join(folder_path, file_name), bbox_inches = 'tight', dpi = 300)  
    plt.close()

    if save_field:
        # Save the field at time t_i to a txt file
        txt_file_name = 'd_λ=' + str(1/config._lambda) + '_w_λ=' + str(config.w/config._lambda)+'_' + str(t_i) + '_carpet.txt'
        savetxt(os.path.join(folder_path, txt_file_name), field[t_i], delimiter=',')



def create_video_from_images(images_path, output_name, fps=30):
    if not os.path.exists(os.path.dirname(images_path)):
        os.makedirs(os.path.dirname(images_path))
        
    command = f'ffmpeg -framerate {fps} -pattern_type glob -i "{images_path}/*.png" -c:v libx264 -pix_fmt yuv420p "{images_path}/{output_name}"'
    os.system(command)

    return print("Video has been done.")