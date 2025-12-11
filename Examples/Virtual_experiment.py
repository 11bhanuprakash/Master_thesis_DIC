import muDIC as dic
from muDIC import vlab
import numpy as np
import matplotlib.pyplot as plt

image_shape = (2000,2000)
speckle_image = vlab.rosta_speckle(
    image_shape,
    dot_size=4,
    density=0.32,
    smoothness=2.0
)
F = np.array([[1.1,.0], [0., 1.0]], dtype=np.float64)

image_deformer = vlab.imageDeformer_from_defGrad(F)

downsampler = vlab.Downsampler(image_shape=image_shape,
    factor=4,
    fill=0.8,
    pixel_offset_stddev=0.1
)

downsampled_speckle = downsampler(speckle_image)

noise_injector = vlab.noise_injector("gaussian", sigma=.1)

n = 10 

image_stack = vlab.SyntheticImageGenerator(speckle_image=speckle_image,
    image_deformer=image_deformer,
    downsampler=downsampler,
    noise_injector=noise_injector,
    n=n
)

image_five = image_stack(5)

# Plot original speckle image
plt.figure(figsize=(8, 8))
plt.title("Original Speckle Image")
plt.imshow(speckle_image, cmap='gray')
plt.axis('off')
plt.show()

# Plot deformed + downsampled + noisy image (image_five)
plt.figure(figsize=(8, 8))
plt.title("Synthetic Image (Frame 5)")
plt.imshow(image_five, cmap='gray')
plt.axis('off')
plt.show()