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
        self.acceleration_eval = None
        
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
        arho = np.zeros_like(x)  # artificial density
        p = np.zeros_like(x)  # pressure
        cs = np.ones_like(x) * 10.0  # speed of sound
        ax = np.zeros_like(x)  # acceleration x
        ay = np.zeros_like(x)  # acceleration y
        az = np.zeros_like(x)  # acceleration z
        u = np.zeros_like(x)  # velocity x
        v = np.zeros_like(x)  # velocity y
        w = np.zeros_like(x)  # velocity z
        
        # Create particle array
        self.particles = get_particle_array(
            name='fluid',
            x=x, y=y, z=z,
            m=m, h=h, rho=rho,
            arho=arho, p=p, cs=cs,
            ax=ax, ay=ay, az=az,
            u=u, v=v, w=w
        )
        
        # Setup equations
        equations = [
            Group(
                equations=[
                    ContinuityEquation(dest='fluid', sources=['fluid']),
                    TaitEOS(dest='fluid', sources=None, rho0=1000, c0=10.0, gamma=7.0),
                    XSPHCorrection(dest='fluid', sources=['fluid'], eps=0.5)
                ]
            )
        ]
        
        # Create NNPS object
        nnps = LinkedListNNPS(dim=3, particles=[self.particles])
        
        # Manually create acceleration evaluation
        from pysph.sph.acceleration_eval import AccelerationEval
        kernel = None  # Will use default kernel
        self.acceleration_eval = AccelerationEval([self.particles], equations, kernel=kernel)
        
        # Setup integrator
        self.integrator = EPECIntegrator(fluid=WCSPHStep())
        self.integrator.set_nnps(nnps)
        self.integrator.set_equations(equations)
        self.integrator.setup([self.particles])
        
        # Initialize solver-like functionality
        self.dt = 0.0001
        self.current_time = 0.0
    
    def step(self):
        """Manual implementation of time stepping"""
        # Compute accelerations
        self.acceleration_eval.compute(self.current_time, self.dt)
        
        # Integrate
        self.integrator.step(self.current_time, self.dt)
        
        # Update time
        self.current_time += self.dt
        
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