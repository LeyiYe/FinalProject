from pysph.base.kernels import CubicSpline
from pysph.solver.application import Application
from pysph.solver.solver import Solver
from pysph.sph.integrator import EPECIntegrator
from pysph.sph.integrator_step import WCSPHStep, SolidMechStep
from pysph.sph.rigid_body import RK2StepRigidBody
from pysph.sph.equation import Group
from pysph.sph.basic_equations import (
    ContinuityEquation, XSPHCorrection, SummationDensity
)

from pysph.sph.solid_mech.basic import (
    HookesDeviatoricStressRate,
    EnergyEquationWithStress,
    MonaghanArtificialStress
)

from pysph.sph.wc.basic import TaitEOS, MomentumEquation
from pysph.base.utils import get_particle_array_rigid_body
from pysph.sph.solid_mech.basic import get_particle_array_elastic_dynamics
from pysph.tools import geometry as G
from pysph.sph.wc.gtvf import VelocityGradient
import numpy as np

# Material properties for rubber-like material
DENSITY = 1000.0  # kg/m^3
STIFFNESS = 1e6    # Pa (Young's modulus, 1MPa is typical for soft rubber)
# VISCOSITY = 0.1    # Viscosity coefficient
ALPHA = 0.3       # Artificial viscosity coefficient
BETA = 0.0         # Second artificial viscosity coefficient
GAMMA = 7.0        # Tait EOS exponent
# YIELD_STRESS = 100 # Yield stress for plasticity

# Simulation parameters
DT = 1e-4
TFINAL = 5.0
DIM = 3

# Domain and object dimensions
BOX_WIDTH = 2.0
BOX_HEIGHT = 2.0
BOX_DEPTH = 2.0
PLATFORM_HEIGHT = 0.1
OBJECT_WIDTH = 0.1
OBJECT_HEIGHT = 0.1
OBJECT_DEPTH = 0.1
GRIPPER_WIDTH = 0.3
GRIPPER_HEIGHT = 0.3
GRIPPER_SPEED = 0.5

class DeformableObjectWithGrippers(Application):
    def __init__(self):
        super().__init__()
        self.particle_arrays = {}

    def add_user_options(self, group):
        group.add_argument(
            "--nx", action="store", type=int, dest="nx", default=20,
            help="Number of particles along x direction in object."
        )
    
    def consume_user_options(self):
        self.nx = self.options.nx
        self.ny = int(self.nx * OBJECT_HEIGHT / OBJECT_WIDTH)
        self.nz = int(self.nx * OBJECT_DEPTH / OBJECT_WIDTH)
        self.hdx = 1.2
        self.dx = OBJECT_WIDTH / self.nx
        self.particle_mass = DENSITY * self.dx**3
    
    def create_particles(self): 
        # First calculate the reference speed of sound
        c0 = np.sqrt(STIFFNESS/DENSITY)
        
        # Create the deformable object
        x, y, z = G.get_3d_block(
            dx=self.dx, 
            length=OBJECT_WIDTH, 
            height=OBJECT_HEIGHT,
            depth=OBJECT_DEPTH,
            center=[0, PLATFORM_HEIGHT + OBJECT_HEIGHT/2, 0]
        )
        
        # Add some random perturbation
        x += np.random.uniform(-self.dx/4, self.dx/4, len(x))
        y += np.random.uniform(-self.dx/4, self.dx/4, len(y))
        z += np.random.uniform(-self.dx/4, self.dx/4, len(z))
        
        # Create elastic object particle array
        object_pa = get_particle_array_elastic_dynamics(
            constants={
                'E': STIFFNESS,
                'nu': 0.3,
                'rho_ref': DENSITY,

            },
            name='object',
            x=x, y=y, z=z,
            u=np.zeros_like(x),
            v=np.zeros_like(x),
            w=np.zeros_like(x),
            rho=np.ones_like(x)*DENSITY,
            m=np.ones_like(x)*self.particle_mass,
            h=np.ones_like(x)*self.dx*self.hdx,
            # Stress tensor components
            s00=np.zeros_like(x),
            s01=np.zeros_like(x),
            s02=np.zeros_like(x),
            s11=np.zeros_like(x),
            s12=np.zeros_like(x),
            s22=np.zeros_like(x),
            s10=np.zeros_like(x),
            s20=np.zeros_like(x),
            s21=np.zeros_like(x),
            # Reference states
            s000=np.zeros_like(x),
            s010=np.zeros_like(x),
            s020=np.zeros_like(x),
            s110=np.zeros_like(x),
            s120=np.zeros_like(x),
            s220=np.zeros_like(x),
            e=np.zeros_like(x),
            e0=np.zeros_like(x),
            # Time step properties
            dt_cfl=np.zeros_like(x),
            dt_force=np.zeros_like(x),
            # Reference states
            rho0=np.ones_like(x)*DENSITY,
            u0=np.zeros_like(x),
            v0=np.zeros_like(x),
            w0=np.zeros_like(x)
        )
        props = ['v00','v01','v02','v10','v11','v12','v20','v21','v22',
            'as00','as01','as02','as11','as12','as22']
        
        for prop in props:
            object_pa.add_property(prop)
            object_pa.get(prop)[:] = 0.0

        object_pa.add_property('uhat')
        object_pa.add_property('vhat')
        object_pa.add_property('what')
        object_pa.add_property('gradvhat', stride=9)
        object_pa.uhat[:] = object_pa.u[:]
        object_pa.vhat[:] = object_pa.v[:]
        object_pa.what[:] = object_pa.w[:]
        object_pa.gradvhat[:] = 0.0

        # Create platform particles
        platform_x, platform_y, platform_z = G.get_3d_block(
            dx=self.dx,
            length=BOX_WIDTH,
            height=PLATFORM_HEIGHT,
            depth=BOX_DEPTH,
            center=[0, PLATFORM_HEIGHT/2, 0]
        )
        
        platform_pa = get_particle_array_rigid_body(
            name='platform',
            x=platform_x, y=platform_y, z=platform_z,
            h=np.ones_like(platform_x) * self.hdx * self.dx,
            m=np.ones_like(platform_x) * self.particle_mass,
            rho=np.ones_like(platform_x) * DENSITY * 100,
            cs=np.ones_like(platform_x) * c0 * 10,
            # Reference states
            rho0=np.ones_like(platform_x) * DENSITY * 100,
            u0=np.zeros_like(platform_x),
            v0=np.zeros_like(platform_x),
            w0=np.zeros_like(platform_x),
            # Time step properties
            dt_cfl=np.zeros_like(platform_x),
            dt_force=np.zeros_like(platform_x)
        )
        # Create gripper particles (left and right)
        left_gripper_x, left_gripper_y, left_gripper_z = G.get_3d_block(
            dx=self.dx,
            length=GRIPPER_WIDTH,
            height=GRIPPER_HEIGHT,
            depth=GRIPPER_WIDTH,
            center=[-BOX_WIDTH/2 + GRIPPER_WIDTH/2, PLATFORM_HEIGHT + GRIPPER_HEIGHT/2, 0]
        )
        
        left_gripper_pa = get_particle_array_rigid_body(
            name='left_gripper',
            x=left_gripper_x, y=left_gripper_y, z=left_gripper_z,
            h=np.ones_like(left_gripper_x) * self.hdx * self.dx,
            m=np.ones_like(left_gripper_x) * self.particle_mass * 10,
            rho=np.ones_like(left_gripper_x) * DENSITY * 10,
            cs=np.ones_like(left_gripper_x) * c0 * 10,
            # Reference states
            rho0=np.ones_like(left_gripper_x) * DENSITY * 10,
            u0=np.zeros_like(left_gripper_x),
            v0=np.zeros_like(left_gripper_x),
            w0=np.zeros_like(left_gripper_x),
            # Time step properties
            dt_cfl=np.zeros_like(left_gripper_x),
            dt_force=np.zeros_like(left_gripper_x)
        )
        right_gripper_x, right_gripper_y, right_gripper_z = G.get_3d_block(
            dx=self.dx,
            length=GRIPPER_WIDTH,
            height=GRIPPER_HEIGHT,
            depth=GRIPPER_WIDTH,
            center=[BOX_WIDTH/2 - GRIPPER_WIDTH/2, PLATFORM_HEIGHT + GRIPPER_HEIGHT/2, 0]
        )
        
        right_gripper_pa = get_particle_array_rigid_body(
            name='right_gripper',
            x=right_gripper_x, y=right_gripper_y, z=right_gripper_z,
            h=np.ones_like(right_gripper_x) * self.hdx * self.dx,
            m=np.ones_like(right_gripper_x) * self.particle_mass * 10,
            rho=np.ones_like(right_gripper_x) * DENSITY * 10,
            cs=np.ones_like(right_gripper_x) * c0 * 10,
            # Reference states
            rho0=np.ones_like(right_gripper_x) * DENSITY * 10,
            u0=np.zeros_like(right_gripper_x),
            v0=np.zeros_like(right_gripper_x),
            w0=np.zeros_like(right_gripper_x),
            # Time step properties
            dt_cfl=np.zeros_like(right_gripper_x),
            dt_force=np.zeros_like(right_gripper_x))
        
        for pa in [platform_pa, left_gripper_pa, right_gripper_pa]:
            pa.add_property('uhat')
            pa.add_property('vhat')
            pa.add_property('what')
            pa.uhat[:] = pa.u[:]
            pa.vhat[:] = pa.v[:]
            pa.what[:] = pa.w[:]
        
        self.particle_arrays['object'] = object_pa
        self.particle_arrays['platform'] = platform_pa
        self.particle_arrays['left_gripper'] = left_gripper_pa
        self.particle_arrays['right_gripper'] = right_gripper_pa

        return list(self.particle_arrays.values())
    

    def create_solver(self):
        kernel = CubicSpline(dim=DIM)
        
        integrator = EPECIntegrator(
            object=SolidMechStep(),  # For deformable object
            platform=RK2StepRigidBody(),  # For rigid platform
            left_gripper=RK2StepRigidBody(),  # For rigid grippers
            right_gripper=RK2StepRigidBody()
        )
        
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
            # Density summation
            Group(equations=[
                SummationDensity(dest='object', sources=['object'])
            ], real=False),
            
            # Stress and energy equations for rubber
            Group(equations=[
                HookesDeviatoricStressRate(
                    dest='object',
                    sources=['object']  
                ),
                EnergyEquationWithStress(
                    dest='object',
                    sources=['object'],
                    alpha=ALPHA,
                    beta=BETA,
                ),
                MonaghanArtificialStress(
                    dest='object',
                    sources=['object']
                )
            ], real=True),
            
            # Momentum equation with corrected gravity
            Group(equations=[
                MomentumEquation(
                    dest='object',
                    sources=['object', 'platform', 'left_gripper', 'right_gripper'],
                    c0=np.sqrt(STIFFNESS/DENSITY),
                    alpha=ALPHA,
                    beta=BETA,
                    gx=0.0,  # Changed from gz to gx for correct direction
                    gy=-9.81,  # Standard gravity in y-direction
                    gz=0.0,
                    tensile_correction=True
                )
            ], real=True)
        ]
        return equations


    # def create_equations(self):
        # equations = [
        # Group(
        #     equations=[
        #         # This calculates velocity gradients (du, dv, dw)
        #         VelocityGradient(
        #             dest='object', 
        #             sources=['object', 'platform', 'left_gripper', 'right_gripper'],
        #             dim = 3
        #         ),
        #     ],
        #     real=True
        # ),

        #     # Density summation for object
        #     Group(
        #         equations=[
        #             SummationDensity(dest='object', sources=['object', 'platform', 'left_gripper', 'right_gripper'])
        #         ],
        #         real=False
        #     ),
            
        #     # Tait equation of state for object
        #     Group(
        #         equations=[
        #             TaitEOS(
        #                 dest='object', sources=None, 
        #                 rho0=DENSITY, c0=np.sqrt(STIFFNESS/DENSITY), gamma=GAMMA
        #             )
        #         ],
        #         real=False
        #     ),
            
        #     # Momentum equation with artificial viscosity
        #     Group(
        #         equations=[
        #             ContinuityEquation(
        #                 dest='object', 
        #                 sources=['object', 'platform', 'left_gripper', 'right_gripper']
        #             ),
        #             MomentumEquation(
        #                 dest='object', 
        #                 sources=['object', 'platform', 'left_gripper', 'right_gripper'],
        #                 alpha=ALPHA, beta=BETA, gy=-9.81, c0=np.sqrt(STIFFNESS/DENSITY)
        #             ),
        #             XSPHCorrection(
        #                 dest='object', 
        #                 sources=['object'], 
        #                 eps=0.5
        #             )
        #         ],
        #         real=True
        #     )
        # ]
        # return equations
    
    def pre_step(self, solver):
        current_time = solver.t
        
        # Control gripper movement
        left_gripper = self.particle_arrays['left_gripper']
        right_gripper = self.particle_arrays['right_gripper']
        
        # First phase: Close grippers (0-1s)
        if current_time < 1.0:
            left_gripper.x[:] += GRIPPER_SPEED * DT
            right_gripper.x[:] -= GRIPPER_SPEED * DT
        
        # Second phase: Lift grippers (1-3s)
        elif current_time < 3.0:
            left_gripper.y[:] += GRIPPER_SPEED * DT
            right_gripper.y[:] += GRIPPER_SPEED * DT
            left_gripper.z[:] += GRIPPER_SPEED * DT 
            right_gripper.z[:] += GRIPPER_SPEED * DT
        
        # Update velocities for visualization
        left_gripper.u[:] = GRIPPER_SPEED if current_time < 1.0 else 0.0
        left_gripper.v[:] = 0.0 if current_time < 1.0 else GRIPPER_SPEED
        right_gripper.u[:] = -GRIPPER_SPEED if current_time < 1.0 else 0.0
        right_gripper.v[:] = 0.0 if current_time < 1.0 else GRIPPER_SPEED
        left_gripper.w[:] = GRIPPER_SPEED if current_time < 3.0 else 0.0
        right_gripper.w[:] = GRIPPER_SPEED if current_time < 3.0 else 0.0
        
    
    def post_step(self, solver):
        pass
    
    def post_process(self):
        pass

if __name__ == '__main__':
    app = DeformableObjectWithGrippers()
    app.run()