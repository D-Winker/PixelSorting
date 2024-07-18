# Pixel Sorting
#
# A project to play around with pixel sorting.
#
# Daniel Winker, July 17, 2024

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

image_path = 'balloons.jpg'
#parameter_list = [{"sorting": "rows", "criteria": "rgb_mean", "filter": {"range": [0,256]}, "selection": "rgb"}]
parameter_list = [{"sorting": "rows", "criteria": "rgb_median", "filter": {"range": [0,256]}, "selection": "rgb"}]


def sort_pixels(pixels, criteria=None, selection=None):
    if criteria == "rgb_mean":
        sorting_criteria = np.mean(pixels, axis=1)
    elif criteria == "rgb_median":
        sorting_criteria = np.median(pixels, axis=1)

    selected = np.zeros(pixels.shape)
    if "r" in selection:
        selected[:,0] += pixels[np.argsort(sorting_criteria), 0]
    if "g" in selection:
        selected[:,1] += pixels[np.argsort(sorting_criteria), 1]
    if "b" in selection:
        selected[:,2] += pixels[np.argsort(sorting_criteria), 2]
    
    return selected


image = Image.open(image_path)

# Convert the image to a numpy array
image_array = np.array(image)
output_image = np.zeros(image_array.shape)

for parameters in parameter_list:
    # Create a temporary image. It will be populated with this set of sorted values.
    temp_image = np.zeros(image_array.shape)

    # Apply any filters to the image:
    if "range" in parameters["filter"]:
        low = parameters["filter"]["range"][0]
        high = parameters["filter"]["range"][1]
        temp_image[(low < image_array) & (image_array < high)] = image_array[(low < image_array) & (image_array < high)]
    
    else:
        temp_image = deepcopy(image_array)

    # Process the image per the sorting rule
    if parameters["sorting"] == "rows":
        temp_image = [sort_pixels(temp_image[row, :, :], parameters["criteria"], parameters["selection"]) for row in range(temp_image.shape[0])]

    output_image += temp_image

# Normalize the output image and ensure the values are the correct datatype
output_image = (output_image / len(parameter_list)).astype(int)

# Display the image
plt.imshow(output_image)
plt.axis('off')  # Hide the axis
plt.show()


"""
Sorting options
- Rows (sort each line, left to right)
- Colums (sort each column, top to bottom)
- Diagonal (sort each diagonal)
- Left to right, top to bottom, restart at the beginning of the next line (scan lines)
- Left to right, top to bottom, continue at the end of the next line (zig zag)
- Diagonal, sort all pixels from the corner out
- Spiral inwards (square, or circle)
- Spiral outwards (square, or circle)
- Rotated 0, 90, 180, 270
- Mirrored

Sorting Criteria
- RGB Mean
- RGB Median
- Hue
- Luminance
- Saturation
- R 
- G 
- B 

Selectable Parameters
- R 
- G 
- B

Selection Options
- Range (min, max)
- Fibonacci
"""