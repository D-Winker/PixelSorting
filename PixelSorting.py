# Pixel Sorting
#
# A project to play around with pixel sorting.
#
# Daniel Winker, July 25, 2024
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

image_path = 'balloons.jpg'
parameter_list = [{"selection": "rows", "criteria": "rgb_mean", "filter": {"range": [25,225]}, "channels": "rgb"},
                  {"selection": "rows", "criteria": "", "filter": {"range": [0,25]}, "channels": "rgb"},
                  {"selection": "rows", "criteria": "", "filter": {"range": [225,256]}, "channels": "rgb"}]

# If there are multiple sets of parameters, so multiple outputs, how should they be merged?
# Only pixels that pass through the chosen filter will be merged; others will "pass through" unaffected
# Options are: mean, min, max
merge_method = {"r": "mean", "g": "mean", "b": "mean"}

def show_img(_img):
    plt.imshow(_img.astype(int))
    plt.axis('off')  # Hide the axis
    plt.show()


def get_sorting_keys(_values=None, _criteria=None):
    if _criteria == "rgb_mean":
        sorting_keys = np.mean(_values, axis=1)
    elif _criteria == "rgb_median":
        sorting_keys = np.median(_values, axis=1)
    else:
        sorting_keys = np.arange(0, _values.shape[0])
    
    return sorting_keys


def get_filter(_values, _filter=None):
    if "range" in _filter:
        low = parameters["filter"]["range"][0]
        high = parameters["filter"]["range"][1]
        _fltr = (low <= _values) & (_values < high)
    else:
        _fltr = (-1 < _values)  # Select everything
    
    return _fltr


def sort_pixels(_values, _keys, _reverse=False):
    """
    Sorts the given 1D array _pixels based on the parallel 
    array of keys, according to the given criteria.
    """
    _values = _values[np.argsort(_keys)]
    
    return _values


image = Image.open(image_path)

# Convert the image to a numpy array
image_array = np.array(image)
output_list = []
mask_list = []  # These are used when merging the resulting outputs

if False:
    # Replace the loaded image with a simple test image
    image_array = np.zeros((10,10,3))
    for c in range(image_array.shape[1]):
        image_array[:,c,0] = (image_array.shape[1] - c - 1) * int(255 / image_array.shape[1])  # R
        image_array[:,c,1] = (image_array.shape[1] - c - 1) * int(255 / image_array.shape[1])  # G
        image_array[:,c,2] = (image_array.shape[1] - c - 1) * int(255 / image_array.shape[1])  # B
    #image_array = np.stack((image_array, np.zeros(image_array.shape), np.zeros(image_array.shape)), axis=2)
    show_img(image_array)

for parameters in parameter_list:
    # Create a temporary image. It will be populated with this set of sorted values.
    temp_image = np.zeros(image_array.shape)
    temp_mask = np.full(image_array.shape, False)

    # Process the image per the selection rule
    if parameters["selection"] == "rows":
        for row in range(image_array.shape[0]):
            # Get the pixels going in
            px_in = image_array[row, :, :]

            # Parse the sorting criteria to give each pixel a key that can be sorted on
            sorting_keys = get_sorting_keys(_values=px_in, _criteria=parameters["criteria"])

            # Sort the selected values and put them back in their appropriate places
            for index, channel in enumerate(["r", "g", "b"]):
                if channel in parameters["channels"]:
                    filter = get_filter(_values=px_in[:,index], _filter=parameters["filter"])
                    temp_mask[row, :, index] = filter
                    temp_image[row, filter, index] = sort_pixels(_values=px_in[filter,index], _keys=sorting_keys[filter])

    mask_list.append(temp_mask)
    output_list.append(temp_image)

# Use the output images and masks to create a masked array
# For a numpy masked array, "True" means "it is True that this value is masked out"
masked_arr = np.ma.array(data=output_list, mask=np.invert(np.asarray(mask_list)))

# Merge the generated images into one
output_image = np.zeros(image_array.shape)
for channel, method in merge_method.items():
    chnl = {"r": 0, "g": 1, "b": 2}[channel]
    if method == "mean":
        output_image[:,:,chnl] += np.ma.getdata(masked_arr[:,:,:,chnl].mean(axis=0))
        #output_image[:,:,chnl] += np.mean(np.asarray(output_list)[:,:,:,chnl], axis=0)
    elif method == "max":
        output_image[:,:,chnl] += np.ma.getdata(masked_arr[:,:,:,chnl].max(axis=0))
        #output_image[:,:,chnl] += np.max(np.asarray(output_list)[:,:,:,chnl], axis=0)
    elif method == "min":
        output_image[:,:,chnl] += np.ma.getdata(masked_arr[:,:,:,chnl].min(axis=0))
        #output_image[:,:,chnl] += np.min(np.asarray(output_list)[:,:,:,chnl], axis=0)

show_img(output_image)


"""
Selection options
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
- Fibonacci
- High pass filter
- Low pass filter
- Square

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
- Contrast
"""