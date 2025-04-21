from pysph.base.kernels import CubicSpline
from pysph.solver.solver import Solver
from pysph.solver.application import Application
from pysph.sph.integrator import EPECIntegrator, PECIntegrator
from pysph.sph.integrator_step import WCSPHStep
from pysph.sph.equation import Group
from pysph.sph.basic_equations import (
    ContinuityEquation, XSPHCorrection
)
from pysph.sph.wc.basic import (
    TaitEOS, MomentumEquation
)
from pysph.sph.solid_mech.basic import (
    HookesDeviatoricStressRate,get_particle_array_elastic_dynamics,
    MomentumEquationWithStress, MonaghanArtificialStress
)
import numpy as np

class DeformableObjectSim(Application):

    def create_particles(self):
        dx = 0.01  # Particle spacing (1mm)
        object_size = 0.05  # 5cm cube
    
    # Create smaller grid
        x, y, z = np.mgrid[
            -object_size/2:object_size/2:dx, 
            -object_size/2:object_size/2:dx, 
            -object_size/2:object_size/2:dx
        ]
        x = x.ravel()
        y = y.ravel()
        z = z.ravel()

        # # Filter out some particles for performance if needed
        # mask = (x**2 + y**2 + z**2) <= (object_size/2)**2
        # x = x[mask]
        # y = y[mask]
        # z = z[mask]

        density = 1000  # kg/m³
        particle_mass = (dx**3) * density

        print(f"Created {len(x)} particles")
        
        # Create particle array with elastic dynamics properties
        particles = get_particle_array_elastic_dynamics(
            constants={
                'E': 1e5,       # Young's modulus (Pa)
                'nu': 0.3,     # Poisson's ratio
                'rho_ref': density # Reference density (kg/m³)
            },
            name='object',
            x=x, y=y, z=z,
            u=np.zeros_like(x),
            v=np.zeros_like(x),
            w=np.zeros_like(x),
            rho=np.ones_like(x)*density,
            m=np.ones_like(x)*(dx**3)*particle_mass,
            h=np.ones_like(x)*dx*1.2,
            p=np.zeros_like(x),
            # Initialize stress tensor components to zero
            s00=np.zeros_like(x),
            s01=np.zeros_like(x),
            s02=np.zeros_like(x),
            s11=np.zeros_like(x),
            s12=np.zeros_like(x),
            s22=np.zeros_like(x)
        )
        return particles


    def create_equations(self):
        equations = [
            Group(equations=[
                TaitEOS(
                    dest='object', sources=None, 
                    rho0=1000.0, c0=100.0, gamma=7.0  # Higher sound speed
                ),
                MomentumEquationWithStress(
                    dest='object', sources=['object'],
                    alpha=0.2, beta=0.2  # Higher viscosity
                ),
                MonaghanArtificialStress(
                    dest='object', sources=['object'],
                    eps=0.5  # Increased artificial stress
                )
            ], real=True)
        ]
        return equations


    def create_solver(self):

        # Use EPECIntegrator for elastic dynamics
        # dt = time step
        integrator = EPECIntegrator(elastic=WCSPHStep())
        kernel = CubicSpline(dim=3)
        solver = Solver(dim=3, integrator=integrator, kernel=kernel,
                tf=1.0, dt=1e-4, adaptive_timestep=True,
                cfl=0.5, output_at_times=[1e-1, 1.0])
        
        return solver