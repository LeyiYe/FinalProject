# pysph_simulation.py
import numpy as np
from pysph.base.utils import get_particle_array
from pysph.sph.equation import Group
from pysph.sph.basic_equations import ContinuityEquation, XSPHCorrection
from pysph.sph.wc.basic import TaitEOS
from pysph.base.nnps import LinkedListNNPS
from pysph.sph.acceleration_eval import AccelerationEval
from pysph.base.kernels import CubicSpline

class DeformableObjectSimulation:
    def __init__(self):
        self.particles = None
        self.nnps = None
        self.equations = None
        self.kernel = None
        self.setup_simulation()
        
    def setup_simulation(self):
        """Initialize a 3D deformable rectangle using SPH"""
        # Particle spacing - can be larger since FleX handles interactions
        dx = 0.05  
        
        # Create a 3D grid of particles (0.5m x 0.5m x 0.2m)
        x = np.arange(-0.25, 0.25, dx)
        y = np.arange(-0.25, 0.25, dx)
        z = np.arange(0, 0.2, dx)
        x, y, z = np.meshgrid(x, y, z)
        x = x.ravel()
        y = y.ravel()
        z = z.ravel()
        
        # Particle properties - simplified since FleX handles physics
        m = np.ones_like(x) * dx * dx * dx * 1000  # mass (kg)
        h = np.ones_like(x) * dx * 1.2  # smoothing length
        rho = np.ones_like(x) * 1000  # density (kg/m^3)
        p = np.zeros_like(x)  # pressure
        cs = np.ones_like(x) * 10.0  # speed of sound
        u = np.zeros_like(x)  # velocity x
        v = np.zeros_like(x)  # velocity y
        w = np.zeros_like(x)  # velocity z
        
        # Add properties needed for FleX integration
        ax = np.zeros_like(x)  # acceleration x
        ay = np.zeros_like(x)  # acceleration y
        az = np.zeros_like(x)  # acceleration z
        arho = np.zeros_like(x)  # density rate
        au = np.zeros_like(x)  # velocity rate x
        av = np.zeros_like(x)  # velocity rate y
        aw = np.zeros_like(x)  # velocity rate z
        
        # Create particle array with all required properties
        self.particles = get_particle_array(
            name='fluid',
            x=x, y=y, z=z,
            m=m, h=h, rho=rho,
            p=p, cs=cs,
            u=u, v=v, w=w,
            ax=ax, ay=ay, az=az,
            arho=arho, au=au, av=av, aw=aw
        )
        
        # Setup equations - simplified since FleX handles most physics
        self.equations = [
            Group(
                equations=[
                    # Basic equations to maintain particle structure
                    XSPHCorrection(dest='fluid', sources=['fluid'], eps=0.5)
                ]
            )
        ]
        
        # Create kernel
        self.kernel = CubicSpline(dim=3)
        
        # Create NNPS object
        self.nnps = LinkedListNNPS(dim=3, particles=[self.particles])
        self.nnps.update()
        
        # Initialize time - can use larger timestep with FleX
        self.dt = 0.001  # Increased from 0.0001
        self.time = 0.0
    
    def step(self):
        """Simplified stepping since FleX handles physics"""
        # Only basic position updates needed
        self.particles.x[:] += self.particles.u[:] * self.dt
        self.particles.y[:] += self.particles.v[:] * self.dt
        self.particles.z[:] += self.particles.w[:] * self.dt
        
        # Update NNPS (neighbor lists)
        self.nnps.update()
        
        # Update time
        self.time += self.dt
        
        return {
            'x': self.particles.x.copy(),
            'y': self.particles.y.copy(),
            'z': self.particles.z.copy(),
            'u': self.particles.u.copy(),
            'v': self.particles.v.copy(),
            'w': self.particles.w.copy()
        }
    
    def get_initial_state(self):
        """Return initial particle positions"""
        return {
            'x': self.particles.x.copy(),
            'y': self.particles.y.copy(),
            'z': self.particles.z.copy(),
            'u': self.particles.u.copy(),
            'v': self.particles.v.copy(),
            'w': self.particles.w.copy()
        }

if __name__ == "__main__":
    sim = DeformableObjectSimulation()
    print(f"Initial particle count: {len(sim.particles.x)}")
    state = sim.step()
    print(f"After one step - first particle position: {state['x'][0]:.3f}, {state['y'][0]:.3f}, {state['z'][0]:.3f}")