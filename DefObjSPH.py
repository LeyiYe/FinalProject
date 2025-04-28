import numpy as np
from mayavi import mlab
from pysph.base.kernels import CubicSpline
from pysph.solver.solver import Solver
from pysph.solver.application import Application
from pysph.sph.integrator import EPECIntegrator
from pysph.sph.integrator_step import SolidMechStep
from pysph.sph.equation import Group
from pysph.sph.wc.basic import TaitEOSHGCorrection
from pysph.sph.basic_equations import XSPHCorrection
from pysph.sph.solid_mech.basic import (
    HookesDeviatoricStressRate, get_particle_array_elastic_dynamics,
    MomentumEquationWithStress, MonaghanArtificialStress
)
from pysph.base.nnps import LinkedListNNPS

class GripperFSM:
    """Finite State Machine to control gripper motion"""
    def __init__(self):
        self.state = "open"
        self.time = 0
        self.close_time = 1.0
        self.hold_time = 2.0
        self.open_time = 3.0
        
    def update(self, dt):
        self.time += dt
        if self.state == "open" and self.time >= self.close_time:
            self.state = "closing"
        elif self.state == "closing" and self.time >= self.hold_time:
            self.state = "closed"
        elif self.state == "closed" and self.time >= self.open_time:
            self.state = "opening"
        elif self.state == "opening" and self.time >= self.open_time + 1.0:
            self.state = "open"
            self.time = 0
            
    def get_finger_position(self, original_pos, max_displacement=0.02):
        if self.state == "open":
            return original_pos
        elif self.state == "closing":
            progress = (self.time - self.close_time) / (self.hold_time - self.close_time)
            return original_pos - max_displacement * progress * np.sign(original_pos)
        elif self.state == "closed":
            return original_pos - max_displacement * np.sign(original_pos)
        elif self.state == "opening":
            progress = (self.time - self.open_time) / 1.0
            return original_pos - max_displacement * (1 - progress) * np.sign(original_pos)

class ElasticDeformableWithGripper(Application):
    def __init__(self, particle_radius=0.02):
        self.particle_radius = particle_radius
        self.object_size = 0.05
        self.kernel = CubicSpline(dim=3)
        self.particle_arrays = []
        self.solver = None
        self.nnps = None
        self.integrator = None
        self.dt = 1e-4
        self.time = 0.0
        self.gripper_fsm = GripperFSM()
        
        # Visualization
        self.mlab_fig = mlab.figure(size=(800, 600), bgcolor=(1, 1, 1))
        self.object_plot = None
        self.gripper_plot = None
        self.text = None
        super().__init__()
        
    def create_particles(self):
        # Deformable rubber-like object
        dx = self.particle_radius
        x, y, z = np.mgrid[
            -self.object_size/2:self.object_size/2:dx, 
            -self.object_size/2:self.object_size/2:dx, 
            -self.object_size/2:self.object_size/2:dx
        ]
        x = x.ravel()
        y = y.ravel()
        z = z.ravel() + 0.1  # Position above gripper

        # Rubber properties (E=1MPa, nu~0.45 for nearly incompressible)
        density = 1200  # kg/m³
        object_pa = get_particle_array_elastic_dynamics(
            constants={
                'E': 1e6,       # 1 MPa - typical for soft rubber
                'nu': 0.45,     # Nearly incompressible
                'rho_ref': density
            },
            name='object',
            x=x, y=y, z=z,
            u=np.zeros_like(x),
            v=np.zeros_like(x),
            w=np.zeros_like(x),
            rho=np.ones_like(x)*density,
            m=np.ones_like(x)*(dx**3)*density,
            h=np.ones_like(x)*dx*1.2,
            p=np.zeros_like(x),
            s00=np.zeros_like(x),
            s01=np.zeros_like(x),
            s02=np.zeros_like(x),
            s11=np.zeros_like(x),
            s12=np.zeros_like(x),
            s22=np.zeros_like(x)
        )
        
        # Panda gripper (simplified)
        # Base (fixed)
        base_x, base_y, base_z = np.mgrid[
            -0.03:0.03:dx*2,
            -0.03:0.03:dx*2,
            -0.01:0.01:dx*2
        ]
        # Fingers (movable)
        left_finger = np.mgrid[
            0.02:0.04:dx,
            -0.01:0.01:dx,
            0.01:0.05:dx
        ]
        right_finger = np.mgrid[
            -0.04:-0.02:dx,
            -0.01:0.01:dx,
            0.01:0.05:dx
        ]
        
        gripper_x = np.concatenate([base_x.ravel(), left_finger[0].ravel(), right_finger[0].ravel()])
        gripper_y = np.concatenate([base_y.ravel(), left_finger[1].ravel(), right_finger[1].ravel()])
        gripper_z = np.concatenate([base_z.ravel(), left_finger[2].ravel(), right_finger[2].ravel()])
        
        gripper_pa = get_particle_array_elastic_dynamics(
            name='gripper',
            x=gripper_x, y=gripper_y, z=gripper_z,
            u=np.zeros_like(gripper_x),
            v=np.zeros_like(gripper_x),
            w=np.zeros_like(gripper_x),
            rho=np.ones_like(gripper_x)*8000,  # Steel density
            m=np.ones_like(gripper_x)*(dx**3)*8000,
            h=np.ones_like(gripper_x)*dx*1.2,
            p=np.zeros_like(gripper_x),
            fixed=np.ones_like(gripper_x, dtype=int),  # Mostly fixed
            s00=np.zeros_like(gripper_x),
            s11=np.zeros_like(gripper_x),
            s22=np.zeros_like(gripper_x)
        )
        gripper_pa.add_property('original_x', data=gripper_x.copy())
        self.particle_arrays = [object_pa, gripper_pa]
        return self.particle_arrays
    
    def create_equations(self):
        equations = [
            # Equation of state (for numerical stability)
            Group(equations=[
                TaitEOSHGCorrection(
                    dest='object', sources=None,
                    rho0=1200.0, c0=10.0, gamma=7.0  # c0=10 gives softer EOS
                ),
            ], real=True),
            
            # Elastic stress and momentum
            Group(equations=[
                HookesDeviatoricStressRate(dest='object', sources=['object']),
                MomentumEquationWithStress(
                    dest='object', 
                    sources=['object', 'gripper']
                ),
                MonaghanArtificialStress(dest='object', sources=['object'], eps=0.3),
                XSPHCorrection(dest='object', sources=['object'], eps=0.5)
            ], real=True),
            
            # Gripper is mostly rigid
            Group(equations=[
                MomentumEquationWithStress(dest='gripper', sources=['object']),
            ], real=False)
        ]
        return equations
    
    def create_solver(self):
        self.integrator = EPECIntegrator(object=SolidMechStep())
        self.solver = Solver(
            dim=3,
            integrator=self.integrator,
            kernel=self.kernel,
            dt=self.dt,
            tf=4.0,  # 4 second simulation
            adaptive_timestep=True,
            cfl=0.2
        )
        equations = self.create_equations()
        self.nnps = LinkedListNNPS(
            dim=3,
            particles=self.particle_arrays,
            radius_scale=self.kernel.radius_scale,
            cache=True
        )
        self.solver.setup(
            particles=self.particle_arrays,
            equations=equations,
            kernel=self.kernel,
            nnps=self.nnps
        )
        return self.solver
    
    def setup_visualization(self):
        mlab.clf()
        obj = self.particle_arrays[0]
        grip = self.particle_arrays[1]
        
        # Object (blue with transparency)
        self.object_plot = mlab.points3d(
            obj.x, obj.y, obj.z,
            color=(0.2, 0.4, 1.0),
            scale_factor=self.particle_radius*1.8,
            opacity=0.7,
            resolution=12
        )
        
        # Gripper (red)
        self.gripper_plot = mlab.points3d(
            grip.x, grip.y, grip.z,
            color=(1, 0, 0),
            scale_factor=self.particle_radius*2,
            opacity=1.0,
            resolution=12
        )
        
        self.text = mlab.text(0.01, 0.9, f"Time: {self.time:.2f}s\nState: {self.gripper_fsm.state}", 
                           width=0.2, color=(0, 0, 0))
        mlab.view(azimuth=45, elevation=60, distance=0.4)
        mlab.orientation_axes()
    
    def update_visualization(self):
        obj = self.particle_arrays[0]
        grip = self.particle_arrays[1]
        
        self.object_plot.mlab_source.set(x=obj.x, y=obj.y, z=obj.z)
        self.gripper_plot.mlab_source.set(x=grip.x, y=grip.y, z=grip.z)
        self.text.text = f"Time: {self.time:.2f}s\nState: {self.gripper_fsm.state}"
        mlab.draw()
        mlab.process_ui_events()
    
    def pre_step(self, solver):
        self.gripper_fsm.update(self.dt)
        gripper_pa = self.particle_arrays[1]
        
        # Move fingers
        is_finger = (gripper_pa.x > 0.015) | (gripper_pa.x < -0.015)
        if np.any(is_finger):
            original_x = gripper_pa.original_x[is_finger]
            new_x = self.gripper_fsm.get_finger_position(original_x)
            gripper_pa.x[is_finger] = new_x
            gripper_pa.u[is_finger] = 0.0  # Zero velocity
    
    def post_step(self, solver):
        if self.time % 0.05 < self.dt:  # Update viz every 0.05s
            self.update_visualization()
    
    def run(self):
        self.create_particles()
        self.create_solver()
        self.setup_visualization()
        
        while self.solver.time < self.solver.tf:
            self.pre_step(self.solver)
            self.solver.step()
            self.post_step(self.solver)
            self.time = self.solver.time
            
            if not mlab.get_engine().scenes:
                print("Visualization closed - stopping")
                break
        
        mlab.show()

if __name__ == '__main__':
    app = ElasticDeformableWithGripper(particle_radius=0.02)
    app.run()