import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Simulation import make_gaussian
import time

# the plane wave that is the input:
E_input_slm = make_gaussian(500,fwhm=100)

# The set SLM (in 8-bit greyvalues)

fig, (ax1, ax3) = plt.subplots(1, 2,figsize=[8,5])

def generate_double_pattern(shape, phases_half1, phases_half2, phase_offset):
    width, height = shape
    half_width = width // 2

    # Generate the first half of the image with phase offset
    half1 = [[int(((i + phase_offset) / (half_width / phases_half1)) * 256) % 256 for _ in range(half_width)] for i in range(height)]

    # Generate the second half of the image
    half2 = [[int((i / (half_width / phases_half2)) * 256) % 256 for _ in range(half_width)] for i in range(height)]

    # Combine both halves
    image_array = [row1 + row2 for row1, row2 in zip(half1, half2)]

    # Resize the image to the specified shape
    image_array = [row[:width] for row in image_array[:height]]

    return np.array(image_array)

SLM_pattern = generate_double_pattern(list(E_input_slm.shape), 4, -10, 0)

E_field_slm = E_input_slm * np.exp(1j * ((SLM_pattern / 256) * 2 * np.pi))

E_field_sample = np.array([])

rnd = np.random.rand(10, 10)
rnd_norm = rnd - np.mean(rnd)

E_field_slm_F = np.fft.fft2(E_field_slm)

img1 = ax1.imshow(SLM_pattern)
plt.colorbar(img1, ax=ax1)
ax1.set_title('SLM pattern')

img3 = ax3.imshow(abs(np.fft.fftshift(E_field_slm_F**2)))
plt.colorbar(img3, ax=ax3)
ax3.set_title('Intensity of light at sample')

def update(frame):
    SLM_pattern = generate_double_pattern(list(E_input_slm.shape), frame+1, -0.001, 0)
    E_field_slm = E_input_slm * np.exp(1j * ((SLM_pattern / 256) * 2 * np.pi))
    E_field_slm_F = np.fft.fft2(E_field_slm)

    img1.set_array(SLM_pattern)
    img3.set_array(abs(np.fft.fftshift(E_field_slm_F**2)))

# Create the animation
animation = FuncAnimation(fig, update, frames=range(20), interval=200)

# Save the animation as a GIF
animation.save('animation.gif', writer='pillow')

# Display the animation
plt.show()