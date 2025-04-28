from pysph.base.kernels import CubicSpline
from pysph.solver.application import Application
from pysph.sph.rigid_body import (BodyForce, RigidBodyCollision, 
                                  RigidBodyMoments, RigidBodyMotion,
                                  RK2StepRigidBody)
from pysph.sph.integrator import EPECIntegrator

from pysph.tools.geometry import get_3d_block, get_3d_sphere, rotate

# Create basic shapes and combine them to form a panda hand
def create_panda_hand(dx=0.05):
    # Palm (main body)
    palm = get_3d_block(dx, length=0.3, height=0.2, width=0.15)
    
    # Fingers (simplified as cylinders)
    fingers = []
    for i in range(5):  # 5 fingers
        finger = get_3d_block(dx, length=0.15, height=0.05, width=0.05)
        # Position each finger
        finger.x += 0.15 + i*0.03
        finger.y += 0.05 if i%2 else -0.05
        fingers.append(finger)
    
    # Combine all parts
    hand = palm
    for finger in fingers:
        hand.add_particles(finger)
    
    return hand


class PandaHandSimulation(Application):
    def initialize(self):
        self.dx = 0.05
        self.hdx = 1.2
        self.ro = 1000.0  # density
        self.rigid_body_mass = 1.0
        
    def create_particles(self):
        # Create panda hand particles
        hand = create_panda_hand(self.dx)
        
        # Set rigid body properties
        hand.add_property('body_id', type='int', data=1)  # all same body
        hand.add_property('m', type='float', data=self.rigid_body_mass/len(hand.x))
        hand.add_property('rho', type='float', data=self.ro)
        
        return [hand]
    
    def create_solver(self):
        kernel = CubicSpline(dim=3)
        
        integrator = EPECIntegrator(hand=RK2StepRigidBody())
        
        from pysph.solver.solver import Solver
        solver = Solver(kernel=kernel, dim=3, integrator=integrator,
                       dt=1e-4, tf=1.0, adaptive_timestep=True)
        return solver
    
    def create_equations(self):
        equations = [
            BodyForce(dest='hand', sources=None),
            RigidBodyCollision(
                dest='hand', sources=['hand'], k=1.0, d=0.1, eta=0.1, kt=0.1
            ),
            RigidBodyMoments(dest='hand', sources=None),
            RigidBodyMotion(dest='hand', sources=None),
        ]
        return equations
    
from pysph.tools import pysph_viewer

if __name__ == '__main__':
    app = PandaHandSimulation()
    app.run()
    pysph_viewer.viewer.show(app.solver)
