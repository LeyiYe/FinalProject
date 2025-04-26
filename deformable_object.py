from pysph.base.kernels import CubicSpline
from pysph.solver.solver import Solver
from pysph.solver.application import Application
from pysph.sph.integrator import EPECIntegrator, PECIntegrator
from pysph.sph.integrator_step import WCSPHStep, SolidMechStep
from pysph.sph.equation import Group
from pysph.sph.equation import Equation

from pysph.sph.wc.basic import TaitEOSHGCorrection

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

class CohesiveForce(Equation):
    def __init__(self, dest, sources, k=1e6):
        self.k = k  # Cohesion stiffness
        super().__init__(dest, sources)

    def initialize(self, d_idx, d_m, d_rho, d_h, d_cohesion):
        d_cohesion[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_m, s_m, d_rho, s_rho, 
             d_h, s_h, d_cohesion, s_cohesion, 
             DWIJ, RIJ, XIJ):
        # Cubic spline cohesion force
        r = RIJ/d_h[d_idx]
        if r < 1.0:
            W = 1 - 1.5*r**2 + 0.75*r**3
        elif r < 2.0:
            W = 0.25*(2-r)**3
        else:
            W = 0.0
            
        d_cohesion[d_idx] += self.k * W * s_m[s_idx]/s_rho[s_idx]

class DeformableObjectSim(Application):
    def __init__(self, particle_radius=0.01):
        self.particle_radius = particle_radius
        self.object_size = 0.05  # 5cm cube
        self.kernel = CubicSpline(dim=3)  # Add this line
        self.particle_array = None
        super().__init__()
        

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

        density = 1000  # kg/m³
        particle_mass = (dx**3) * density

        print(f"Created {len(x)} particles")
        
        # Create particle array with elastic dynamics properties
        self.particle_array = get_particle_array_elastic_dynamics(
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
            s22=np.zeros_like(x),
            cohesion=np.zeros_like(x)
        )

        return [self.particle_array]

    def get_particles(self):
        """Helper method to get the single particle array"""
        return self.particle_array

    def create_equations(self):
        equations = [
            Group(equations=[
                #Stronger EOS for incompressible materials
                TaitEOSHGCorrection(
                    dest='object', sources=None,
                    rho0=1000.0, c0=140.0, gamma=7.0  # Higher sound speed
                ),

            ], real=True),

                # Elastic stress formulation
            Group(
                equations=[
                    HookesDeviatoricStressRate(
                        dest='object', sources=['object']
                    ),
                    MomentumEquationWithStress(
                        dest='object', sources=['object']  # Higher viscosity
                    )
                ], real =True
            ),

                # Enhanced cohesion
            Group(
                equations=[
                    CohesiveForce(
                        dest='object', sources=['object'],
                        k=1e6  # Higher cohesion stiffness
                    )], real=True
            ),
            
                #Artificial stress and viscosity
            Group(
                equations=[
                    MonaghanArtificialStress(dest='object', 
                                    sources=['object'],
                                    eps=1.0),
                    XSPHCorrection(
                        dest='object', 
                        sources=['object'],
                        eps=0.5  # Smother motion
                    )
                ], real=True
            )
        ]
        return equations


    def create_solver(self):
        # Use EPECIntegrator for elastic dynamics
        from pysph.base.nnps import LinkedListNNPS
        from pysph.sph.acceleration_eval import AccelerationEval

        integrator = EPECIntegrator(object=SolidMechStep())

        solver = Solver(dim=3, 
                        integrator=integrator, 
                        kernel=self.kernel,
                        dt=1e-4,  # Initial time step
                        tf=1.0,  # Final time
                        adaptive_timestep=True,  # Enable adaptive time stepping
                        cfl=0.1  # Courant-Friedrichs-Lewy condition
                        )
        
        particles = self.particles_array
        equations = self.create_equations()

        print(f"Particle x range: {min(particles[0].x)} to {max(particles[0].x)}")
        print(f"Particle y range: {min(particles[0].y)} to {max(particles[0].y)}")
        print(f"Particle z range: {min(particles[0].z)} to {max(particles[0].z)}")

        domain_size = 0.1

        nnps = LinkedListNNPS(
            dim=3, 
            particles=particles, 
            radius_scale=self.kernel.radius_scale,
            cache = True,
            domain = ((-domain_size/2, -domain_size/2, -domain_size/2),  # min x,y,z
            (domain_size/2, domain_size/2, domain_size/2) # max x,y,z
            )
        )

        solver.setup(
            particles=particles,
            equations=equations,
            kernel=self.kernel,
            nnps=nnps
        )
        return solver