import numpy as np
from pysph.base.utils import get_particle_array

def create_soft_cube(size=0.1, resolution=10, density=1000):
    """Create a cubic deformable object."""
    x, y, z = np.mgrid[
        -size/2:size/2:resolution*1j,
        -size/2:size/2:resolution*1j,
        -size/2:size/2:resolution*1j
    ]
    positions = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    
    return get_particle_array(
        name="object",
        x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
        m=np.ones(len(positions)) * (density * (size/resolution)**3),
        h=np.ones(len(positions)) * (size/resolution),
        rho=np.ones(len(positions)) * density,
    )

def create_soft_sphere(radius=0.05, resolution=10, density=1000):
    """Create a spherical deformable object."""
    # ... (similar logic for spheres)