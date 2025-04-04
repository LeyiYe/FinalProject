import numpy as np
from pysph.base.utils import get_particle_array
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the dimensions of the rectangle
length = 2.0  # along x-axis
width = 1.0   # along y-axis
height = 0.5  # along z-axis

# Define particle spacing (adjust for resolution)
dx = 0.05

# Generate particle positions in a 3D grid
x = np.arange(0, length, dx)
y = np.arange(0, width, dx)
z = np.arange(0, height, dx)

xx, yy, zz = np.meshgrid(x, y, z)
positions = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

# Create a PySPH particle array
particles = get_particle_array(
    name="rectangle",
    x=positions[:, 0],
    y=positions[:, 1],
    z=positions[:, 2],
)

# Save the particle positions to a file
np.savetxt("rectangle_particles.txt", positions)

# Visualize the particles in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of particles
ax.scatter(
    positions[:, 0],  # x-coordinates
    positions[:, 1],  # y-coordinates
    positions[:, 2],  # z-coordinates
    s=10,             # particle size
    c='blue',         # color
    alpha=0.5,        # transparency
    marker='o'        # marker style
)

# Labels and title
ax.set_xlabel('X-axis (Length)')
ax.set_ylabel('Y-axis (Width)')
ax.set_zlabel('Z-axis (Height)')
ax.set_title('3D Rectangle Model (Particle Representation)')

# Equal aspect ratio (to avoid distortion)
ax.set_box_aspect([length, width, height])

plt.tight_layout()
plt.show()