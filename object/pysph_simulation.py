# pysph_simulation.py
import numpy as np
from pysph.base.utils import get_particle_array
from pysph.solver.solver import Solver
from pysph.sph.integrator import EPECIntegrator
from pysph.sph.integrator_step import WCSPHStep
from pysph.sph.equation import Group
from pysph.sph.basic_equations import ContinuityEquation, XSPHCorrection
from pysph.sph.wc.basic import TaitEOS
from pysph.base.nnps import LinkedListNNPS

class DeformableObjectSimulation:
    def __init__(self):
        self.solver = None
        self.particles = None
        self.setup_simulation()
        
    def setup_simulation(self):
        """Initialize a 3D deformable rectangle using SPH"""
        # Particle spacing
        dx = 0.05
        
        # Create a 3D grid of particles (0.5m x 0.5m x 0.2m)
        x = np.arange(-0.25, 0.25, dx)
        y = np.arange(-0.25, 0.25, dx)
        z = np.arange(0, 0.2, dx)
        x, y, z = np.meshgrid(x, y, z)
        x = x.ravel()
        y = y.ravel()
        z = z.ravel()
        
        # Particle properties
        m = np.ones_like(x) * dx * dx * dx * 1000  # mass (kg)
        h = np.ones_like(x) * dx * 1.2  # smoothing length
        rho = np.ones_like(x) * 1000  # density (kg/m^3)
        
        # Create particle array
        self.particles = get_particle_array(
            name='fluid',
            x=x, y=y, z=z,
            m=m, h=h, rho=rho
        )
        
        # Setup solver
        integrator = EPECIntegrator(fluid=WCSPHStep())
        self.solver = Solver(
            dim=3,
            integrator=integrator,
            dt=0.0001,
            tf=10.0
        )
        
        nnps = LinkedListNNPS(dim=3, particles=[self.particles])

        # Setup equations
        self.solver.setup(
            [self.particles],
            [
                Group(
                    equations=[
                        ContinuityEquation(dest='fluid', sources=['fluid']),
                        TaitEOS(dest='fluid', sources=None, rho0=1000, c0=10.0, gamma=7.0),
                        XSPHCorrection(dest='fluid', sources=['fluid'])
                    ]
                )
            ]
            nnps=nnps
        )
    
    def step(self):
        """Advance the simulation by one timestep"""
        self.solver.step(1)
        return {
            'x': self.particles.x.copy(),
            'y': self.particles.y.copy(),
            'z': self.particles.z.copy()
        }
    
    def get_initial_state(self):
        """Return initial particle positions"""
        return {
            'x': self.particles.x.copy(),
            'y': self.particles.y.copy(),
            'z': self.particles.z.copy()
        }

if __name__ == "__main__":
    # Test the simulation
    sim = DeformableObjectSimulation()
    print(f"Initial particle count: {len(sim.particles.x)}")
    state = sim.step()
    print(f"After one step - first particle position: {state['x'][0]:.3f}, {state['y'][0]:.3f}, {state['z'][0]:.3f}")