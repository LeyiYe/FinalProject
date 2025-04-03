from pysph.base.kernels import CubicSpline
from pysph.solver.application import Application
from pysph.sph.scheme import WCSPHScheme
from pysph.sph.equation import Group
from pysph.sph.basic_equations import ContinuityEquation, MomentumEquation
import numpy as np

class DeformableObject(Application):
    def create_particles(self):
        # Create a 3D grid of particles
        spacing = 0.05  # Particle spacing
        nx, ny, nz = 10, 10, 10  # Cube dimensions (10x10x10)

        x, y, z = np.mgrid[0:nx, 0:ny, 0:nz] * spacing
        x, y, z = x.ravel(), y.ravel(), z.ravel()

        # Particle properties
        mass = np.ones_like(x) * 0.01
        rho = np.ones_like(x) * 1000  # Density

        # Create a PySPH particle array
        from pysph.base.utils import get_particle_array
        particles = get_particle_array(name='deformable', x=x, y=y, z=z, m=mass, rho=rho)

        return [particles]

    def create_scheme(self):
        return WCSPHScheme(
            fluids=['deformable'],
            dim=3,
            rho0=1000,
            c0=10,
            h=0.1,
            gamma=7,
            alpha=0.1,
            beta=0.1
        )

# Run PySPH to generate particles
app = DeformableObject()
app.configure()
particles = app.create_particles()[0]

# Save particle positions to a file
np.save("sph_particles.npy", np.column_stack((particles.x, particles.y, particles.z)))
