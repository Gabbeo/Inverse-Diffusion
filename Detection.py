import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import img_as_float
from skimage.feature import peak_local_max
from ScoreOptimizer import optimal_f1_score

cell_number = 1250

# Loads the image using Pillow (ImageMagick).
image_file = Image.open("inverse_diffusion_final_" + str(cell_number) + ".tiff")
image = img_as_float(image_file) * 255

# Loads text file containing true cell positions.
file_object = open(str(cell_number) + "_Noi3.txt")

# Reads true cell positions from text file.
true_coordinates = np.zeros([cell_number, 2])
for row in range(cell_number):
    stringTuple = file_object.readline().split(",")
    intTuple = [int(stringTuple[1]) - 1, int(stringTuple[0].rstrip()) - 1]
    true_coordinates[row] = intTuple

# Find local intensity peaks in input image.
local_peaks = peak_local_max(image, min_distance=1, threshold_abs=0)

# Reverses order of x-values and y-values, as they are
# returned in the wrong order from peak_local_max().
calculated_coordinates = np.zeros([len(local_peaks), 2])
for i, item in enumerate(local_peaks):
    calculated_coordinates[i][0] = item[1]
    calculated_coordinates[i][1] = item[0]

# Adds pixel values for calculated coordinates to pseudo-likelihood array.
pseudo_likelihood = np.zeros([len(calculated_coordinates)])
for i, item in enumerate(calculated_coordinates):
    pseudo_likelihood[i] = image[int(item[1])][int(item[0])]

# Runs function to determine F1-score and optimal pixel threshold.
optimal_threshold, f1_score = optimal_f1_score(true_coordinates, calculated_coordinates, pseudo_likelihood, 1)

print("Optimal threshold: " + str(optimal_threshold))
print("F1-score: " + str(f1_score))

# Removes detection points whose pseudo-likelihood is below the optimal threshold.
remove_list = []
for i, item in enumerate(pseudo_likelihood):
    if item < optimal_threshold:
        remove_list.append(i)
calculated_coordinates = np.delete(calculated_coordinates, remove_list, axis=0)

np.savetxt("..\\Matlab Cell Detection\\calculated_coordinates.csv", calculated_coordinates, delimiter=",")
np.savetxt("..\\Matlab Cell Detection\\real_coordinates.csv", true_coordinates, delimiter=",")
np.savetxt("..\\Matlab Cell Detection\\pseudo_likelihood.csv", pseudo_likelihood, delimiter=",")

print("Done!")

# Displays results.
fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
ax = axes.ravel()
original_image_file = Image.open("image_" + str(cell_number) + ".tiff")
orgiginal_image = img_as_float(original_image_file)

ax[0].imshow(orgiginal_image, cmap=plt.cm.gray)
ax[0].axis("off")
ax[0].set_title("Original image")

ax[1].imshow(image, cmap=plt.cm.gray)
ax[1].axis("off")
ax[1].set_title("Processed image")

ax[2].imshow(image, cmap=plt.cm.gray)
ax[2].autoscale(False)
for element in true_coordinates:
    ax[2].plot(element[0], element[1], "bo")
for element in calculated_coordinates:
    ax[2].plot(element[0], element[1], "r+")
ax[2].axis("off")
ax[2].set_title("Cell detections")

fig.tight_layout()

plt.show()
