import numpy as np
from pysph.sph.solid_mech.basic import get_particle_array_elastic_dynamics

def create_soft_cube(size=0.1, resolution=10, density=1200, 
                    youngs_modulus=1e6, poissons_ratio=0.45):
    """Create a cubic deformable object with proper elastic properties.
    
    Args:
        size: Edge length of the cube (meters)
        resolution: Particles per edge
        density: Material density (kg/m³)
        youngs_modulus: Elastic modulus (Pa)
        poissons_ratio: Material Poisson ratio (0.3-0.49)
    """
    # Create particle positions
    x, y, z = np.mgrid[
        -size/2:size/2:resolution*1j,
        -size/2:size/2:resolution*1j,
        -size/2:size/2:resolution*1j
    ]
    positions = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    
    # Particle spacing and smoothing length
    spacing = size / resolution
    h = np.ones(len(positions)) * spacing * 1.2  # Typical h = 1.2*dx
    
    # Create specialized particle array for elastic dynamics
    return get_particle_array_elastic_dynamics(
        name="object",
        x=positions[:,0], y=positions[:,1], z=positions[:,2],
        m=np.ones(len(positions)) * (density * spacing**3),
        h=h,
        rho=np.ones(len(positions)) * density,
        constants={
            'E': youngs_modulus,       # Young's modulus
            'nu': poissons_ratio,      # Poisson's ratio
            'rho_ref': density,        # Reference density
            'c0_ref': np.sqrt(youngs_modulus/density)  # Speed of sound
        }
    )

def create_soft_sphere(radius=0.05, resolution=10, density=1200,
                      youngs_modulus=1e6, poissons_ratio=0.45):
    """Create a spherical deformable object."""
    # Generate points within sphere
    theta, phi = np.mgrid[0:np.pi:resolution*1j, 0:2*np.pi:resolution*1j]
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    
    positions = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    
    # Approximate spacing and smoothing length
    spacing = 2*np.pi*radius / resolution
    h = np.ones(len(positions)) * spacing * 1.2
    
    return get_particle_array_elastic_dynamics(
        name="object",
        x=positions[:,0], y=positions[:,1], z=positions[:,2],
        m=np.ones(len(positions)) * (density * spacing**3),
        h=h,
        rho=np.ones(len(positions)) * density,
        constants={
            'E': youngs_modulus,
            'nu': poissons_ratio,
            'rho_ref': density,
            'c0_ref': np.sqrt(youngs_modulus/density)
        }
    )