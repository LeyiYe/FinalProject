import numpy as np
from pysph.base.utils import get_particle_array

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

# Save the particle positions to a CSV file with headers
header = "x,y,z"
np.savetxt("rectangle_particles.csv", 
           positions, 
           delimiter=",", 
           header=header, 
           comments="",
           fmt='%.5f')  # Save with 5 decimal places