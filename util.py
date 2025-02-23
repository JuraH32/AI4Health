import numpy as np

def invert_image(image):
    new_image = np.max(image) - image
    return new_image - np.min(new_image)