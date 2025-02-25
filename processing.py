import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt

def resize_downsample(image, size=(224, 224)):
    # Use cv2.resize for faster resizing on numpy arrays.
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

def average_pooling_downsample(image, size=(224, 224)):
    kernel_size = image.shape[2] // size[0]
    stride = kernel_size
    return F.avg_pool2d(image, kernel_size=kernel_size, stride=stride)

def max_pooling_downsample(image, size=(224, 224)):
    kernel_size = image.shape[2] // size[0]
    stride = kernel_size
    return F.max_pool2d(image, kernel_size=kernel_size, stride=stride)

def find_breast(image):
    new_image = image.copy()
    if len(image.shape) == 3:
        new_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.dtype != np.uint8:
        new_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    blurred_image = cv2.GaussianBlur(new_image, (5, 5), 0)
    binary_mask = cv2.threshold(blurred_image, 20, 255, cv2.THRESH_BINARY)[1]
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    x = stats[largest_component, cv2.CC_STAT_LEFT]
    y = stats[largest_component, cv2.CC_STAT_TOP]
    w = stats[largest_component, cv2.CC_STAT_WIDTH]
    h = stats[largest_component, cv2.CC_STAT_HEIGHT]
    return (x, y, w, h)

def invert_image(image):
    new_image = np.max(image) - image
    return new_image - np.min(new_image)

def invert_if_needed(image):
    image = image - np.min(image)
    if np.sum(image == np.max(image)) > np.sum(image == np.min(image)):
        return invert_image(image)
    return image

def remove_borders(image):
    row_mean = np.mean(image, axis=1)
    col_mean = np.mean(image, axis=0)
    max_value = np.max(image) - 5
    row_max_count = np.sum(row_mean >= max_value)
    col_max_count = np.sum(col_mean >= max_value)
    row_min_count = np.sum(row_mean == 0)
    col_min_count = np.sum(col_mean == 0)
    row_start = 0
    row_end = image.shape[0]
    col_start = 0
    col_end = image.shape[1]
    if row_max_count > 0:
        row_indices = np.where(row_mean < max_value)[0]
        row_start = row_indices[0]
        row_end = row_indices[-1]
    if col_max_count > 0:
        col_indices = np.where(col_mean < max_value)[0]
        col_start = col_indices[0]
        col_end = col_indices[-1]
    if row_min_count > 0:
        row_indices = np.where(row_mean > 0)[0]
        row_start = max(row_start, row_indices[0])
        row_end = min(row_end, row_indices[-1])
    if col_min_count > 0:
        col_indices = np.where(col_mean > 0)[0]
        col_start = max(col_start, col_indices[0])
        col_end = min(col_end, col_indices[-1])
    return image[row_start:row_end, col_start:col_end]

# -----------------------------------------------------------
# Transformation wrappers
# -----------------------------------------------------------
class ResizeDownsampleTransform:
    def __init__(self, size=(224, 224)):
        self.size = size

    def __call__(self, image):
        return resize_downsample(image, self.size)

class AveragePoolingDownsampleTransform:
    def __init__(self, size=(224, 224)):
        self.size = size

    def __call__(self, image):
        return average_pooling_downsample(image, self.size)

class MaxPoolingDownsampleTransform:
    def __init__(self, size=(224, 224)):
        self.size = size

    def __call__(self, image):
        return max_pooling_downsample(image, self.size)

class InvertIfNeededTransform:
    def __call__(self, image):
        # Assuming image is a numpy array; if using tensors, convert as needed.
        return invert_if_needed(image)

class RemoveBordersTransform:
    def __call__(self, image):
        return remove_borders(image)

class FindBreastTransform:
    def __call__(self, image):
        """
        Uses the find_breast function to locate the breast,
        and then crops the image to that region.
        """
        x, y, w, h = find_breast(image)
        return image[y:y+h, x:x+w]

# -----------------------------------------------------------
# Compose transformation pipeline
# -----------------------------------------------------------
def get_transform(method="resize", apply_invert=False, apply_remove_borders=False, apply_find=False):
    """
    Args:
        method (str): Downsampling method. Options: 'resize', 'avg', 'max'.
        apply_invert (bool): Whether to invert the image if needed.
        apply_remove_borders (bool): Whether to remove borders.
        apply_find (bool): Whether to crop the image to the breast region.
    Returns:
        A composed transformation.
    """
    transforms_list = []

    if apply_find:
        transforms_list.append(FindBreastTransform())

    if method == "resize":
        transforms_list.append(ResizeDownsampleTransform(size=(224, 224)))
    elif method == "avg":
        transforms_list.append(AveragePoolingDownsampleTransform(size=(224, 224)))
    elif method == "max":
        transforms_list.append(MaxPoolingDownsampleTransform(size=(224, 224)))
    else:
        raise ValueError("Unknown downsampling method.")

    # Optionally apply inversion
    if apply_invert:
        transforms_list.append(InvertIfNeededTransform())

    # Optionally remove borders
    if apply_remove_borders:
        transforms_list.append(RemoveBordersTransform())

    # You can add additional transformations here as needed.
    return transforms.Compose(transforms_list)

# -----------------------------------------------------------
# Example usage
# -----------------------------------------------------------
if __name__ == "__main__":
    # Create a dummy image as a numpy array (grayscale) for demonstration purposes.
    # For a real-case, you may use cv2.imread or other sources to read an image.
    dummy_image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)

    # Create a transformation pipeline
    transform_pipeline = get_transform(method="resize", apply_find=True, apply_invert=True, apply_remove_borders=True)
    transformed_image = transform_pipeline(dummy_image)

    print("Original shape:", dummy_image.shape)
    print("Transformed shape:", transformed_image.shape)

    # Plot original and transformed image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(dummy_image, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("Transformed Image")
    plt.imshow(transformed_image, cmap="gray")
    plt.show()