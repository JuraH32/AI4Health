import torch
import torch.nn.functional as F
from torchvision import transforms

def resize_downsample(image, size=(224, 224)):
    transform = transforms.Resize(size)
    return transform(image)

def average_pooling_downsample(image, kernel_size=2, stride=2):
    return F.avg_pool2d(image, kernel_size=kernel_size, stride=stride)

def max_pooling_downsample(image, kernel_size=2, stride=2):
    return F.max_pool2d(image, kernel_size=kernel_size, stride=stride)

# Example usage
if __name__ == "__main__":
    # Create a dummy image tensor with shape (1, 3, 256, 256)
    dummy_image = torch.randn(1, 3, 256, 256)

    # Test resize downsampling
    resized_image = resize_downsample(dummy_image)
    print(f"Resized image shape: {resized_image.shape}")

    # Test average pooling downsampling
    avg_pooled_image = average_pooling_downsample(dummy_image)
    print(f"Average pooled image shape: {avg_pooled_image.shape}")

    # Test max pooling downsampling
    max_pooled_image = max_pooling_downsample(dummy_image)
    print(f"Max pooled image shape: {max_pooled_image.shape}")