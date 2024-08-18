import matplotlib.pyplot as plt
import numpy as np

# Create weight map with different sigma values
sigma_values = [5, 10, 15]
image_shape = (48, 64)
eye_center = (24, 32)  # Coordinates should match (y, x)

def create_gaussian_weight_map(image_shape, eye_center=(24, 32), sigma=15):
    """
    Creates a Gaussian weight map centered around the eye.
    image_shape: (height, width)
    eye_center: (y, x) coordinates of the eye center
    sigma: Standard deviation of the Gaussian distribution
    """
    y, x = np.meshgrid(np.arange(image_shape[0]), np.arange(image_shape[1]), indexing='ij')
    distance = (x - eye_center[1])**2 + (y - eye_center[0])**2  # Use [1] for x and [0] for y
    weight_map = np.exp(-distance / (2 * sigma**2))
    
    # Normalize the weight map so that it ranges from 0 to 1
    weight_map = weight_map / np.max(weight_map)
    
    # Expand dimensions to match the shape (height, width, channels)
    weight_map = np.expand_dims(weight_map, axis=-1)
    
    return weight_map

for sigma in sigma_values:
    weight_map = create_gaussian_weight_map(image_shape, eye_center, sigma)
    plt.imshow(weight_map.squeeze(), cmap='hot')
    plt.title(f'Gaussian Weight Map with sigma={sigma}')
    plt.colorbar()
    # plt.show()

    plt.savefig(f'gaussian_weight_map_sigma_{sigma}.png')
    plt.close()  # Close the figure to free up memory
