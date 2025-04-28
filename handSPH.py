import numpy as np
from pysph.base.kernels import CubicSpline
from pysph.solver.application import Application
from pysph.sph.rigid_body import (BodyForce, RigidBodyCollision, 
                                  RigidBodyMoments, RigidBodyMotion,
                                  RK2StepRigidBody)
from pysph.sph.integrator import EPECIntegrator
from pysph.tools.geometry import get_3d_block

def create_panda_hand(dx=0.05):  # Increased from 0.02 to 0.05
    """Create a detailed Panda robotic hand model with fingers."""
    particles = {'x': [], 'y': [], 'z': []}
    
    # 1. Create palm
    palm_x, palm_y, palm_z = get_3d_block(
        dx, length=0.15, height=0.1, depth=0.12,
        center=np.array([0.0, 0.0, 0.0])
    )
    particles['x'].append(palm_x)
    particles['y'].append(palm_y)
    particles['z'].append(palm_z)
    
    # 2. Create wrist/base (simplified)
    wrist_x, wrist_y, wrist_z = get_3d_block(
        dx, length=0.1, height=0.08, depth=0.08,
        center=np.array([-0.08, 0.0, 0.0])
    )
    particles['x'].append(wrist_x)
    particles['y'].append(wrist_y)
    particles['z'].append(wrist_z)
    
    # 3. Create fingers (simplified)
    finger_params = [
        # (name, position, length)
        ('index', [0.1, 0.06, 0.0], 0.12),
        ('middle', [0.1, 0.0, 0.0], 0.14),
        ('thumb', [0.05, 0.0, 0.06], 0.1)
    ]
    
    for name, pos, length in finger_params:
        seg_x, seg_y, seg_z = get_3d_block(
            dx, 
            length=length,
            height=0.04,
            depth=0.04,
            center=np.array([
                pos[0] + length/2 if name != 'thumb' else pos[0],
                pos[1],
                pos[2]
            ])
        )
        particles['x'].append(seg_x)
        particles['y'].append(seg_y)
        particles['z'].append(seg_z)
    
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
        self.damping = 0.1      # Added damping
        
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
                damping=self.damping  # Added damping
            ),
            RigidBodyCollision(
                dest='object',
                sources=['hand', 'object'],
                kn=self.kn,
                mu=self.mu,
                en=self.en,
                damping=self.damping  # Added damping
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