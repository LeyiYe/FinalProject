import numpy as np
from pysph.base.utils import get_particle_array
from pysph.tools.geometry import get_3d_block

class SphSphere:
    def __init__(self, config):
        """
        Creates a deformable SPH sphere
        
        Config parameters:
        - radius: float (default 0.5)
        - spacing: float (particle spacing, default 0.1)
        - density: float (default 1000.0)
        - E: float (Young's modulus, default 1e5)
        - nu: float (Poisson's ratio, default 0.3)
        """
        self.config = {
            'radius': 0.5,
            'spacing': 0.1,
            'density': 1000.0,
            'E': 1e5,
            'nu': 0.3,
            **config  # Override defaults with provided config
        }
        self.particles = self._create_particles()

    def _create_particles(self):
        """Generate particle positions for a sphere"""
        dx = self.config['spacing']
        radius = self.config['radius']
        
        # Create a solid block first
        x, y, z = get_3d_block(dx, 2*radius, 2*radius, 2*radius)
        
        # Filter to sphere shape
        dist = np.sqrt(x**2 + y**2 + z**2)
        mask = dist <= radius
        x, y, z = x[mask], y[mask], z[mask]
        
        # Create particle array
        pa = get_particle_array(
            name='sphere',
            x=x, y=y, z=z,
            m=np.ones_like(x) * dx**3 * self.config['density'],
            rho=np.ones_like(x) * self.config['density'],
            h=np.ones_like(x) * dx * 1.2,
            E=np.ones_like(x) * self.config['E'],
            nu=np.ones_like(x) * self.config['nu']
        )
        
        # Additional properties needed for elasticity
        pa.add_property('vm_stress')  # Von Mises stress
        pa.add_property('eps')        # Strain tensor
        pa.add_property('sig')        # Stress tensor
        
        return pa

    def get_particle_array(self):
        """Returns the PySPH particle array"""
        return self.particles