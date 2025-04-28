from pysph.base.kernels import CubicSpline
from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.sph.integrator import EPECIntegrator
from pysph.sph.integrator_step import WCSPHStep
from pysph.sph.equation import Group
from pysph.sph.basic_equations import (
    ContinuityEquation, XSPHCorrection, SummationDensity
)
from pysph.sph.wc.basic import TaitEOS, MomentumEquation
from pysph.base.utils import (get_particle_array, get_particle_array_rigid_body)
from pysph.sph.solid_mech.basic import get_particle_array_elastic_dynamics
from pysph.tools import geometry as G
import numpy as np

# Material properties for rubber-like material
DENSITY = 1000.0  # kg/m^3
STIFFNESS = 1e5    # Bulk modulus
VISCOSITY = 0.1    # Viscosity coefficient
ALPHA = 0.1        # Artificial viscosity coefficient
BETA = 0.0         # Second artificial viscosity coefficient
GAMMA = 7.0        # Tait EOS exponent
YIELD_STRESS = 100 # Yield stress for plasticity

# Simulation parameters
DT = 1e-4
TFINAL = 5.0
DIM = 2

# Domain and object dimensions
BOX_WIDTH = 2.0
BOX_HEIGHT = 2.0
PLATFORM_HEIGHT = 0.1
OBJECT_WIDTH = 0.5
OBJECT_HEIGHT = 0.3
GRIPPER_WIDTH = 0.06
GRIPPER_HEIGHT = 0.05
GRIPPER_SPEED = 0.5

class DeformableObjectWithGrippers(Application):
    def add_user_options(self, group):
        group.add_argument(
            "--nx", action="store", type=int, dest="nx", default=20,
            help="Number of particles along x direction in object."
        )
    
    def consume_user_options(self):
        self.nx = self.options.nx
        self.ny = int(self.nx * OBJECT_HEIGHT / OBJECT_WIDTH)
        self.hdx = 1.2
        self.dx = OBJECT_WIDTH / self.nx
        self.particle_mass = DENSITY * self.dx**2
    
    def create_particles(self): 
        # First calculate the reference speed of sound
        c0 = np.sqrt(STIFFNESS/DENSITY)
        
        # Create the deformable object using get_particle_array_elastic_dynamics
        x, y = G.get_2d_block(
            dx=self.dx, 
            length=OBJECT_WIDTH, 
            height=OBJECT_HEIGHT,
            center=[0, PLATFORM_HEIGHT + OBJECT_HEIGHT/2]
        )
        
        # Add some random perturbation
        x += np.random.uniform(-self.dx/4, self.dx/4, len(x))
        y += np.random.uniform(-self.dx/4, self.dx/4, len(y))
        z = np.zeros_like(x)  # For 2D simulation
        
        particle_mass = (self.dx**3) * DENSITY

        # Create elastic object particle array
        object_pa = get_particle_array_elastic_dynamics(
            constants={
                'E': STIFFNESS,
                'nu': 0.3,
                'rho_ref': DENSITY
            },
            name='object',
            x=x, y=y, z=z,
            u=np.zeros_like(x),
            v=np.zeros_like(x),
            w=np.zeros_like(x),
            rho=np.ones_like(x)*DENSITY,
            m=np.ones_like(x)*particle_mass,
            h=np.ones_like(x)*self.dx*self.hdx,
            p=np.zeros_like(x),
            s00=np.zeros_like(x),
            s01=np.zeros_like(x),
            s02=np.zeros_like(x),
            s11=np.zeros_like(x),
            s12=np.zeros_like(x),
            s22=np.zeros_like(x),
            dt_cfl=np.zeros_like(x),
            dt_force=np.zeros_like(x),
            au=np.zeros_like(x),
            av=np.zeros_like(x),
            aw=np.zeros_like(x),
            cs=np.ones_like(x)*c0,  # Speed of sound

    
        )
        
        # Create platform particles
        platform_x, platform_y = G.get_2d_block(
            dx=self.dx,
            length=BOX_WIDTH,
            height=PLATFORM_HEIGHT,
            center=[0, PLATFORM_HEIGHT/2]
        )
        platform_z = np.zeros_like(platform_x)
        
        platform_pa = get_particle_array_rigid_body(
            name='platform',
            x=platform_x, y=platform_y, z=platform_z,
            h=np.ones_like(platform_x) * self.hdx * self.dx,
            m=np.ones_like(platform_x) * self.particle_mass,
            rho=np.ones_like(platform_x) * DENSITY * 100,
            cs=np.ones_like(platform_x) * c0 * 10,
            rho0=np.ones_like(platform_x) * DENSITY * 100,
            u0=np.zeros_like(platform_x),
            v0=np.zeros_like(platform_x),
            w0=np.zeros_like(platform_x)
        )
        
        # Create gripper particles (left and right)
        left_gripper_x, left_gripper_y = G.get_2d_block(
            dx=self.dx,
            length=GRIPPER_WIDTH,
            height=GRIPPER_HEIGHT,
            center=[-BOX_WIDTH/2 + GRIPPER_WIDTH/2, PLATFORM_HEIGHT + GRIPPER_HEIGHT/2]
        )
        left_gripper_z = np.zeros_like(left_gripper_x)
        
        left_gripper_pa = get_particle_array_rigid_body(
            name='left_gripper',
            x=left_gripper_x, y=left_gripper_y, z=left_gripper_z,
            h=np.ones_like(left_gripper_x) * self.hdx * self.dx,
            m=np.ones_like(left_gripper_x) * self.particle_mass * 10,
            rho=np.ones_like(left_gripper_x) * DENSITY * 10,
            cs=np.ones_like(left_gripper_x) * c0 * 10,    
            rho0=np.ones_like(platform_x) * DENSITY * 100,
            u0=np.zeros_like(platform_x),
            v0=np.zeros_like(platform_x),
            w0=np.zeros_like(platform_x)
        )
        
        right_gripper_x, right_gripper_y = G.get_2d_block(
            dx=self.dx,
            length=GRIPPER_WIDTH,
            height=GRIPPER_HEIGHT,
            center=[BOX_WIDTH/2 - GRIPPER_WIDTH/2, PLATFORM_HEIGHT + GRIPPER_HEIGHT/2]
        )
        right_gripper_z = np.zeros_like(right_gripper_x)
        
        right_gripper_pa = get_particle_array_rigid_body(
            name='right_gripper',
            x=right_gripper_x, y=right_gripper_y, z=right_gripper_z,
            h=np.ones_like(right_gripper_x) * self.hdx * self.dx,
            m=np.ones_like(right_gripper_x) * self.particle_mass * 10,
            rho=np.ones_like(right_gripper_x) * DENSITY * 10,
            cs=np.ones_like(right_gripper_x) * c0 * 10,
            rho0=np.ones_like(platform_x) * DENSITY * 100,
            u0=np.zeros_like(platform_x),
            v0=np.zeros_like(platform_x),
            w0=np.zeros_like(platform_x)
        )
        
        object_pa.add_property('rho0')
        object_pa.add_property('u0')
        object_pa.add_property('v0')
        object_pa.add_property('w0')
        object_pa.rho0[:] = DENSITY
        object_pa.u0[:] = 0.0
        object_pa.v0[:] = 0.0
        object_pa.w0[:] = 0.0

        return [object_pa, platform_pa, left_gripper_pa, right_gripper_pa]
    
    def create_solver(self):
        kernel = CubicSpline(dim=DIM)
        
        integrator = EPECIntegrator(object=WCSPHStep(), platform=WCSPHStep(),
                                  left_gripper=WCSPHStep(), right_gripper=WCSPHStep())
        
        solver = Solver(
            kernel=kernel,
            dim=DIM,
            integrator=integrator,
            dt=DT,
            tf=TFINAL,
            adaptive_timestep=True,
            output_at_times=[0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
        )
        
        return solver
    
    def create_equations(self):
        equations = [
            # Density summation for object
            Group(
                equations=[
                    SummationDensity(dest='object', sources=['object', 'platform', 'left_gripper', 'right_gripper'])
                ],
                real=False
            ),
            
            # Tait equation of state for object
            Group(
                equations=[
                    TaitEOS(
                        dest='object', sources=None, 
                        rho0=DENSITY, c0=np.sqrt(STIFFNESS/DENSITY), gamma=GAMMA
                    )
                ],
                real=False
            ),
            
            # Momentum equation with artificial viscosity
            Group(
                equations=[
                    ContinuityEquation(
                        dest='object', 
                        sources=['object', 'platform', 'left_gripper', 'right_gripper']
                    ),
                    MomentumEquation(
                        dest='object', 
                        sources=['object', 'platform', 'left_gripper', 'right_gripper'],
                        alpha=ALPHA, beta=BETA, gz=-9.81, c0 = np.sqrt(STIFFNESS/DENSITY)
                    ),
                    XSPHCorrection(
                        dest='object', 
                        sources=['object'], 
                        eps=0.5
                    )
                ],
                real=True
            ),
            
            # Platform and grippers are treated as rigid bodies (no equations)
        ]
        
        return equations
    
    def pre_step(self, solver):
        # This gets called before each time step
        current_time = solver.t
        
        # Control gripper movement
        left_gripper = self.particles['left_gripper']
        right_gripper = self.particles['right_gripper']
        
        # First phase: Close grippers
        if current_time < 1.0:
            left_gripper.x[:] += GRIPPER_SPEED * DT
            right_gripper.x[:] -= GRIPPER_SPEED * DT
        # Second phase: Lift grippers
        elif current_time < 3.0:
            left_gripper.y[:] += GRIPPER_SPEED * DT
            right_gripper.y[:] += GRIPPER_SPEED * DT
        
        # Update velocities for visualization
        left_gripper.u[:] = GRIPPER_SPEED if current_time < 1.0 else 0.0
        left_gripper.v[:] = 0.0 if current_time < 1.0 else GRIPPER_SPEED
        right_gripper.u[:] = -GRIPPER_SPEED if current_time < 1.0 else 0.0
        right_gripper.v[:] = 0.0 if current_time < 1.0 else GRIPPER_SPEED
        
        # Simple plasticity model - yield stress
        object_pa = self.particles['object']
        strain = np.sqrt(object_pa.du**2 + object_pa.dv**2) / STIFFNESS
        plastic = strain > YIELD_STRESS
        if np.any(plastic):
            object_pa.u[plastic] *= 0.99  # Dampen velocity for plastic deformation
            object_pa.v[plastic] *= 0.99
    
    def post_step(self, solver):
        # This gets called after each time step
        pass
    
    def post_process(self):
        # This gets called after simulation completes
        pass

if __name__ == '__main__':
    app = DeformableObjectWithGrippers()
    app.run()