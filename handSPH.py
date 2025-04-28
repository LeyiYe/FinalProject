import numpy as np
from pysph.base.kernels import CubicSpline
from pysph.solver.application import Application
from pysph.sph.rigid_body import (BodyForce, RigidBodyCollision, 
                                  RigidBodyMoments, RigidBodyMotion,
                                  RK2StepRigidBody)
from pysph.sph.integrator import EPECIntegrator
from pysph.tools.geometry import get_3d_block

def create_panda_hand(dx=0.05):
    """Create a biologically accurate panda hand with two grippers."""
    # Main palm/arm section
    palm_x, palm_y, palm_z = get_3d_block(
        dx, length=0.4, height=0.15, depth=0.2,
        center=np.array([0.0, 0.0, 0.0])
    )
    
    # Two grippers (modified wrist bones)
    gripper_params = [
        # (x_offset, y_offset, length, height, depth)
        (0.2, 0.1, 0.15, 0.07, 0.07),  # Upper gripper
        (0.2, -0.1, 0.15, 0.07, 0.07)   # Lower gripper
    ]
    
    grippers = []
    for i, (x_off, y_off, l, h, d) in enumerate(gripper_params):
        gx, gy, gz = get_3d_block(
            dx, length=l, height=h, depth=d,
            center=np.array([x_off, y_off, 0.0])
        )
        grippers.append((gx, gy, gz))
    
    # Combine all components
    x = np.concatenate([palm_x] + [g[0] for g in grippers])
    y = np.concatenate([palm_y] + [g[1] for g in grippers])
    z = np.concatenate([palm_z] + [g[2] for g in grippers])
    
    return x, y, z

class PandaHandSimulation(Application):
    def initialize(self):
        # Simulation parameters
        self.dx = 0.05          # Particle spacing
        self.hdx = 1.2          # Ratio of h/dx
        self.ro = 1000.0        # Reference density (kg/m^3)
        self.rigid_body_mass = 2.0  # Total mass of hand (kg)
        self.kn = 1e6           # Normal stiffness for collisions
        self.mu = 0.5           # Friction coefficient (higher for better grip)
        self.en = 0.1           # Low restitution (panda hands are not bouncy)
        
    def create_particles(self):
        from pysph.base.utils import get_particle_array_rigid_body
        
        # Create panda hand particles
        x, y, z = create_panda_hand(self.dx)
        
        # Create rigid body particle array
        hand = get_particle_array_rigid_body(
            x=x, y=y, z=z,
            h=self.hdx*self.dx,
            m=self.rigid_body_mass/len(x),
            rho=self.ro,
            name='hand'
        )
        
        # Set additional rigid body properties
        hand.total_mass[0] = self.rigid_body_mass
        hand.body_id[:] = 1  # All particles belong to body 1
        
        # Create a bamboo stalk for gripping
        bamboo_x, bamboo_y, bamboo_z = get_3d_block(
            self.dx, length=0.05, height=0.8, depth=0.05,
            center=np.array([0.3, 0.0, 0.0])
        )
        bamboo = get_particle_array_rigid_body(
            x=bamboo_x, y=bamboo_y, z=bamboo_z,
            h=self.hdx*self.dx,
            m=0.5/len(bamboo_x),  # Total mass of 0.5 kg
            rho=self.ro,
            name='bamboo'
        )
        bamboo.body_id[:] = 2
        bamboo.total_mass[0] = np.sum(bamboo.m)
        
        return [hand, bamboo]
    
    def create_solver(self):
        kernel = CubicSpline(dim=3)
        integrator = EPECIntegrator(hand=RK2StepRigidBody(), bamboo=RK2StepRigidBody())
        
        from pysph.solver.solver import Solver
        solver = Solver(
            kernel=kernel,
            dim=3,
            integrator=integrator,
            dt=1e-4,
            tf=2.0,
            adaptive_timestep=True
        )
        return solver
    
    def create_equations(self):
        equations = [
            # Gravity force
            BodyForce(dest='hand', sources=None, gy=-9.81),
            BodyForce(dest='bamboo', sources=None, gy=-9.81),
            
            # Rigid body collisions
            RigidBodyCollision(
                dest='hand',
                sources=['hand', 'bamboo'],
                kn=self.kn,
                mu=self.mu,
                en=self.en
            ),
            RigidBodyCollision(
                dest='bamboo',
                sources=['hand', 'bamboo'],
                kn=self.kn,
                mu=self.mu,
                en=self.en
            ),
            
            # Rigid body dynamics
            RigidBodyMoments(dest='hand', sources=None),
            RigidBodyMotion(dest='hand', sources=None),
            RigidBodyMoments(dest='bamboo', sources=None),
            RigidBodyMotion(dest='bamboo', sources=None),
        ]
        return equations

if __name__ == '__main__':
    app = PandaHandSimulation()
    app.run()