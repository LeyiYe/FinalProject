import numpy as np
from pysph.base.kernels import CubicSpline
from pysph.solver.application import Application
from pysph.sph.rigid_body import (BodyForce, RigidBodyCollision, 
                                  RigidBodyMoments, RigidBodyMotion,
                                  RK2StepRigidBody)
from pysph.sph.integrator import EPECIntegrator
from pysph.tools.geometry import get_3d_block, get_3d_hollow_cylinder, get_3d_sphere

def create_panda_hand(dx=0.02):
    """Create a detailed Panda robotic hand model with fingers."""
    particles = {'x': [], 'y': [], 'z': []}
    
    # 1. Create palm (larger and more rounded)
    palm_x, palm_y, palm_z = get_3d_block(
        dx, length=0.15, height=0.1, depth=0.12,
        center=np.array([0.0, 0.0, 0.0])
    )
    particles['x'].append(palm_x)
    particles['y'].append(palm_y)
    particles['z'].append(palm_z)
    
    # 2. Create wrist/base (cylindrical)
    wrist_x, wrist_y, wrist_z = get_3d_hollow_cylinder(
        dx, length=0.1, r=0.06,
        center=np.array([-0.1, 0.0, 0.0])
    )
    particles['x'].append(wrist_x)
    particles['y'].append(wrist_y)
    particles['z'].append(wrist_z)
    
    # 3. Create fingers (more detailed)
    finger_params = [
        # (name, position, length, segments)
        ('index', [0.1, 0.06, 0.0], 0.12, 3),
        ('middle', [0.1, 0.0, 0.0], 0.14, 3),
        ('thumb', [0.05, 0.0, 0.06], 0.1, 2)
    ]
    
    for name, pos, length, segments in finger_params:
        segment_length = length / segments
        for i in range(segments):
            seg_x, seg_y, seg_z = get_3d_block(
                dx*0.8,  # Slightly denser particles for fingers
                length=segment_length,
                height=0.04,
                depth=0.04,
                center=np.array([
                    pos[0] + (i+0.5)*segment_length if name != 'thumb' else pos[0],
                    pos[1],
                    pos[2]
                ])
            )
            particles['x'].append(seg_x)
            particles['y'].append(seg_y)
            particles['z'].append(seg_z)
    
    # 4. Create fingertip spheres for better contact
    for name, pos, length, segments in finger_params:
        tip_x, tip_y, tip_z = get_3d_sphere(
            dx*0.7,
            r=0.03,
            center=np.array([
                pos[0] + length if name != 'thumb' else pos[0] + length*0.7,
                pos[1],
                pos[2]
            ])
        )
        particles['x'].append(tip_x)
        particles['y'].append(tip_y)
        particles['z'].append(tip_z)
    
    # Combine all components
    x = np.concatenate(particles['x'])
    y = np.concatenate(particles['y'])
    z = np.concatenate(particles['z'])
    
    return x, y, z

class PandaHandSimulation(Application):
    def initialize(self):
        # Simulation parameters
        self.dx = 0.02          # Finer particle spacing for more detail
        self.hdx = 1.2          # Ratio of h/dx
        self.ro = 2500.0        # Higher density for metal parts (kg/m^3)
        self.rigid_body_mass = 3.5  # Total mass of hand (kg)
        self.kn = 1e7           # Higher stiffness for rigid robotic parts
        self.mu = 0.8           # Higher friction coefficient for better grip
        self.en = 0.05          # Very low restitution (robotic hands aren't bouncy)
        
    def create_particles(self):
        from pysph.base.utils import get_particle_array_rigid_body
        
        # Create detailed panda hand particles
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
        
        # Create a test object (cylinder shaped)
        obj_x, obj_y, obj_z = get_3d_hollow_cylinder(
            self.dx, length=0.2, r=0.04,
            center=np.array([0.2, 0.0, 0.0])
        )
        obj = get_particle_array_rigid_body(
            x=obj_x, y=obj_y, z=obj_z,
            h=self.hdx*self.dx,
            m=0.3/len(obj_x),  # Total mass of 0.3 kg
            rho=self.ro,
            name='object'
        )
        obj.body_id[:] = 2
        obj.total_mass[0] = np.sum(obj.m)
        
        return [hand, obj]
    
    def create_solver(self):
        kernel = CubicSpline(dim=3)
        integrator = EPECIntegrator(hand=RK2StepRigidBody(), object=RK2StepRigidBody())
        
        from pysph.solver.solver import Solver
        solver = Solver(
            kernel=kernel,
            dim=3,
            integrator=integrator,
            dt=1e-5,  # Smaller timestep for finer particles
            tf=2.0,
            adaptive_timestep=True
        )
        return solver
    
    def create_equations(self):
        equations = [
            # Gravity force
            BodyForce(dest='hand', sources=None, gy=-9.81),
            BodyForce(dest='object', sources=None, gy=-9.81),
            
            # Rigid body collisions
            RigidBodyCollision(
                dest='hand',
                sources=['hand', 'object'],
                kn=self.kn,
                mu=self.mu,
                en=self.en
            ),
            RigidBodyCollision(
                dest='object',
                sources=['hand', 'object'],
                kn=self.kn,
                mu=self.mu,
                en=self.en
            ),
            
            # Rigid body dynamics
            RigidBodyMoments(dest='hand', sources=None),
            RigidBodyMotion(dest='hand', sources=None),
            RigidBodyMoments(dest='object', sources=None),
            RigidBodyMotion(dest='object', sources=None),
        ]
        return equations

if __name__ == '__main__':
    app = PandaHandSimulation()
    app.run()