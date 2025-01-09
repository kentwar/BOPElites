from PIL import Image
import matplotlib.pyplot as plt

# Generate and save matplotlib images
images = []  # List to store images
homedir = '~/PhD_code/BOP-Elites-2022/Tutorials/experiment_data/robotarm/MAP-Elites/50/Test/9090/'
homedir = '~/PhD_code/BOP-Elites-2022/tools/experiment_data/robotarm/BOP_UKD_beta_beta/10/9090/'
exphome = os.path.expanduser(homedir)

for i in range(20):
    # Generate your plot or visualization
    # Append the image to the list
    images.append(Image.open(f"{exphome}frame{i}archive.png"))

    # Clear the plot for the next frame
    plt.clf()

# Save the list of images as an animated GIF
images[0].save("animationBOP.gif", save_all=True, append_images=images[1:], duration=200, loop=0)