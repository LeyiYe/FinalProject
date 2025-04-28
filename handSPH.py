import numpy as np
from pysph.base.kernels import CubicSpline
from pysph.solver.application import Application
from pysph.sph.rigid_body import (BodyForce, RigidBodyCollision, 
                                  RigidBodyMoments, RigidBodyMotion,
                                  RK2StepRigidBody)
from pysph.sph.integrator import EPECIntegrator
from pysph.tools.geometry import get_3d_block

def create_panda_hand(dx=0.05):
    """Create a Panda robotic hand with left and right grippers for grasping."""
    particles = {'x': [], 'y': [], 'z': []}
    
    # 1. Create main body (central rectangular block)
    main_body_x, main_body_y, main_body_z = get_3d_block(
        dx, 
        length=0.15,   # Shorter length for the base
        height=0.1,    # Height of main body
        depth=0.2,     # Wider depth to accommodate side grippers
        center=np.array([0.0, 0.0, 0.0])
    )
    particles['x'].append(main_body_x)
    particles['y'].append(main_body_y)
    particles['z'].append(main_body_z)
    
    # 2. Create two gripper blocks (left and right)
    gripper_params = [
        # (x_offset, z_offset, length, height, depth)
        (0.0, 0.12, 0.2, 0.08, 0.05),  # Right gripper
        (0.0, -0.12, 0.2, 0.08, 0.05)   # Left gripper
    ]
    
    for x_off, z_off, l, h, d in gripper_params:
        gx, gy, gz = get_3d_block(
            dx,
            length=l,
            height=h,
            depth=d,
            center=np.array([x_off, 0.0, z_off])  # Centered vertically (y=0)
        )
        particles['x'].append(gx)
        particles['y'].append(gy)
        particles['z'].append(gz)
    
    # Combine all components
    x = np.concatenate(particles['x'])
    y = np.concatenate(particles['y'])
    z = np.concatenate(particles['z'])
    
    return x, y, z

class PandaHandSimulation(Application):
    def initialize(self):
        # More stable parameters
        self.dx = 0.05          # Coarser spacing for stability
        self.hdx = 1.5          # Larger smoothing length
        self.ro = 2500.0        # Density
        self.rigid_body_mass = 3.5
        self.kn = 1e5           # Reduced stiffness (was 1e7)
        self.mu = 0.5           # Moderate friction
        self.en = 0.1           # Slightly higher restitution
        
    def create_particles(self):
        from pysph.base.utils import get_particle_array_rigid_body
        
        x, y, z = create_panda_hand(self.dx)
        
        hand = get_particle_array_rigid_body(
            x=x, y=y, z=z,
            h=self.hdx*self.dx,
            m=self.rigid_body_mass/len(x),
            rho=self.ro,
            name='hand'
        )
        hand.total_mass[0] = self.rigid_body_mass
        hand.body_id[:] = 1
        
        # Simpler test object
        obj_x, obj_y, obj_z = get_3d_block(
            self.dx, length=0.1, height=0.1, depth=0.1,
            center=np.array([0.2, 0.0, 0.0])
        )
        obj = get_particle_array_rigid_body(
            x=obj_x, y=obj_y, z=obj_z,
            h=self.hdx*self.dx,
            m=0.3/len(obj_x),
            rho=self.ro,
            name='object'
        )
        obj.body_id[:] = 2
        obj.total_mass[0] = np.sum(obj.m)
        
        return [hand, obj]
    
    def create_solver(self):
        kernel = CubicSpline(dim=3)
        integrator = EPECIntegrator(
            hand=RK2StepRigidBody(), 
            object=RK2StepRigidBody()
        )
        
        from pysph.solver.solver import Solver
        solver = Solver(
            kernel=kernel,
            dim=3,
            integrator=integrator,
            dt=1e-4,           # Larger timestep
            tf=1.0,            # Shorter simulation time
            adaptive_timestep=True,
            pfreq=50           # Print progress more frequently
        )
        return solver
    
    def create_equations(self):
        equations = [
            BodyForce(dest='hand', sources=None, gy=-9.81),
            BodyForce(dest='object', sources=None, gy=-9.81),
            
            RigidBodyCollision(
                dest='hand',
                sources=['hand', 'object'],
                kn=self.kn,
                mu=self.mu,
                en=self.en,
            ),
            RigidBodyCollision(
                dest='object',
                sources=['hand', 'object'],
                kn=self.kn,
                mu=self.mu,
                en=self.en,
            ),
            
            RigidBodyMoments(dest='hand', sources=None),
            RigidBodyMotion(dest='hand', sources=None),
            RigidBodyMoments(dest='object', sources=None),
            RigidBodyMotion(dest='object', sources=None),
        ]
        return equations

if __name__ == '__main__':
    app = PandaHandSimulation()
    app.run()