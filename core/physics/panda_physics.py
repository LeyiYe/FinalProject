import numpy as np
from transforms3d.euler import euler2mat
from pysph.solver.application import Application
from pysph.sph.solid_mech.basic import ElasticSolidsScheme, get_particle_array_elastic_dynamics
from pysph.base.utils import get_particle_array
from pysph.base.kernels import CubicSpline
from core.utils.urdf_processor import URDFProcessor
from core.physics.panda_fk import get_fk
from pysph.solver.utils import load

class PandaGraspSimulation(Application):
    def initialize(self):
        """Initialize simulation parameters"""
        # Material properties for deformable object
        self.E = 1e6      # Young's modulus (1 MPa)
        self.nu = 0.45     # Poisson's ratio
        self.rho0 = 1200   # Reference density (kg/m^3)
        
        # Simulation parameters
        self.dx = 0.01     # Particle spacing
        self.hdx = 1.2     # h/dx ratio
        self.dt = 1e-5     # Time step
        self.tf = 1.0      # Final time
        
        # Hand configuration
        self.hand_origin = np.eye(4)
        self.joint_positions = {}

    def add_user_options(self, group):
        """Add command line options for the application"""
        group.add_argument(
            "--urdf-file", action="store", type=str, dest="urdf_file",
            default="franka_description/robots/common/hand.urdf",
            help="Path to Panda hand URDF file"
        )
        group.add_argument(
            "--cube-size", action="store", type=float, dest="cube_size",
            default=0.1, help="Size of deformable cube (m)"
        )
        group.add_argument(
            "--resolution", action="store", type=int, dest="resolution",
            default=15, help="Particles per cube dimension"
        )

    def consume_user_options(self):
        # Load URDF
        self.urdf_processor = URDFProcessor(self.options.urdf_file)
        self.boundaries = self._create_boundaries()

    def create_scheme(self):
        """Create the SPH scheme"""
        return ElasticSolidsScheme(
            elastic_solids=['object'],
            solids=[],
            dim=3,
            artificial_stress_eps=0.3,
            alpha=1.0,
            beta=1.0
        )

    def configure_scheme(self):
        """Configure the scheme"""
        s = self.scheme
        s.configure_solver(
            dt=self.dt,
            tf=self.tf,
            pfreq=100,
            adaptive_timestep=True
        )

    def create_particles(self):
        """Create all particle arrays"""
        # Create deformable cube
        cube = self._create_deformable_cube()
        
        # Create hand boundaries
        boundaries = self._create_boundary_particles()
        
        return [cube] + boundaries

    def _create_deformable_cube(self):
        """Create particle array for deformable cube"""
        dx = self.options.cube_size / (self.options.resolution - 1)
        x, y, z = np.mgrid[
            -self.options.cube_size/2:self.options.cube_size/2:self.options.resolution*1j,
            -self.options.cube_size/2:self.options.cube_size/2:self.options.resolution*1j,
            -self.options.cube_size/2:self.options.cube_size/2:self.options.resolution*1j
        ]
        x = x.ravel()
        y = y.ravel()
        z = z.ravel() + self.options.cube_size  # Position below hand
        
        # Calculate particle properties
        volume = dx**3
        m = np.ones_like(x) * volume * self.rho0
        h = np.ones_like(x) * dx * self.hdx
        
        # Create elastic particle array
        cube = get_particle_array_elastic_dynamics(
            name="object",
            x=x, y=y, z=z,
            m=m, h=h,
            rho=np.ones_like(x)*self.rho0,
            constants={
                'E': self.E,
                'nu': self.nu,
                'rho_ref': self.rho0
            }
        )
        cube.add_property('tag')
        cube.add_property('color')
        cube.color[:] = 0  # Blue for object
        
        return cube

    def _create_boundary_particles(self):
        """Create particle arrays for hand boundaries"""
        boundaries = []
        for name, points in self.boundaries.items():
            arr = get_particle_array(
                name=f'hand_boundary_{name}',
                x=points[:,0], y=points[:,1], z=points[:,2],
                m=np.ones(len(points)) * 0.1,
                h=np.ones(len(points)) * self.dx*self.hdx,
                rho=np.ones(len(points)) * 2000,
                tag=np.ones(len(points)),
                color=np.ones(len(points))
            )
            boundaries.append(arr)
        return boundaries

    def _create_boundaries(self):
        """Convert URDF collision shapes to boundary point clouds"""
        boundaries = {}
        for link_name, link_data in self.urdf_processor.links.items():
            for i, collision in enumerate(link_data['collisions']):
                boundary_name = f"{link_name}_collision_{i}"
                boundaries[boundary_name] = self._create_boundary_from_collision(collision)
        return boundaries

    def _create_boundary_from_collision(self, collision):
        """Generate points for different collision geometries"""
        geo_type = collision['type']
        params = collision['params']
        pos = collision['position']
        rot = collision['rotation']
        
        # Generate base points
        if geo_type == 'box':
            size = list(map(float, params['size'].split()))
            points = np.mgrid[
                -size[0]/2:size[0]/2:5j,
                -size[1]/2:size[1]/2:5j,
                -size[2]/2:size[2]/2:5j
            ].reshape(3,-1).T
            
        elif geo_type == 'sphere':
            radius = float(params['radius'])
            theta, phi = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
            points = np.vstack([
                radius*np.cos(theta)*np.sin(phi),
                radius*np.sin(theta)*np.sin(phi),
                radius*np.cos(phi)
            ]).reshape(3,-1).T
            
        elif geo_type == 'cylinder':
            radius, length = float(params['radius']), float(params['length'])
            theta, h = np.mgrid[0:2*np.pi:10j, -length/2:length/2:5j]
            points = np.vstack([
                radius*np.cos(theta),
                radius*np.sin(theta),
                h
            ]).reshape(3,-1).T
            
        else:  # Fallback for meshes
            points = np.array([[0,0,0]])
        
        # Apply transform
        homog = np.column_stack([points, np.ones(len(points))])
        tf = np.eye(4)
        tf[:3,:3] = rot
        tf[:3,3] = pos
        return (tf @ homog.T).T[:,:3]

    def pre_step(self, solver):
        """Called before each timestep - update hand position"""
        # Update finger positions based on simulation time
        finger_pos = min(0.04, 0.04 * (solver.t / self.tf))
        self.joint_positions = {
            "panda_finger_joint1": -finger_pos,
            "panda_finger_joint2": finger_pos
        }
        
        # Update all boundary particles
        for name, points in self.boundaries.items():
            link_name = '_'.join(name.split('_')[:-2])
            tf = self._get_link_tf(link_name)
            transformed_points = self._apply_transform(points, tf)
            
            # Update solver particles
            arr_name = f'hand_boundary_{name}'
            if arr_name in solver.particles.arrays:
                arr = solver.particles[arr_name]
                arr.x[:], arr.y[:], arr.z[:] = transformed_points.T

    def post_step(self, solver):
        """Called after each timestep - monitor contacts"""
        contacts = self._get_contacts(solver)
        if solver.count % 10 == 0:
            print(f"Time: {solver.t:.4f}, Contacts: {len(contacts)}")

    def _get_contacts(self, solver):
        """Calculate contacts between object and hand"""
        contacts = []
        obj = solver.particles.arrays[0]  # Deformable object
        
        for name in self.boundaries:
            boundary = solver.particles[f'hand_boundary_{name}']
            
            # Calculate distances efficiently
            dx = obj.x[:,None] - boundary.x
            dy = obj.y[:,None] - boundary.y
            dz = obj.z[:,None] - boundary.z
            dists = np.sqrt(dx**2 + dy**2 + dz**2)
            
            # Find contacts within interaction range
            contact_mask = dists < self.scheme.h0
            contact_indices = np.where(contact_mask)
            
            for obj_idx, boundary_idx in zip(*contact_indices):
                contacts.append({
                    'particle_id': obj_idx,
                    'boundary': name,
                    'position': [obj.x[obj_idx], obj.y[obj_idx], obj.z[obj_idx]],
                    'distance': dists[obj_idx, boundary_idx]
                })
        
        return contacts

    def post_process(self, info_fname):
        """Post-process simulation results"""
        if self.rank > 0:
            return
            
        # Load results and analyze
        data = load(info_fname)
        
        # Example analysis - plot final state
        self._visualize_final_state(data['solver_data']['particles'])

    def _visualize_final_state(self, particles):
        """Visualize the final simulation state"""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot object particles
        obj = particles.arrays[0]
        ax.scatter(obj.x, obj.y, obj.z, c='b', s=20, alpha=0.6, label='Object')
        
        # Plot hand particles
        for name in self.boundaries:
            hand = particles[f'hand_boundary_{name}']
            ax.scatter(hand.x, hand.y, hand.z, c='r', s=10, alpha=0.8, label='Hand')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Final Simulation State')
        ax.legend()
        plt.tight_layout()
        plt.show()

    def _get_link_tf(self, link_name):
        """Get current transform for a robot link"""
        joint_array = np.array([self.joint_positions.get(f'joint_{i}', 0.0)
                              for i in range(16)])
        mode = 'left' if 'left' in link_name.lower() else \
               'right' if 'right' in link_name.lower() else 'mid'
        return get_fk(joint_array, self.hand_origin, mode)

    def _apply_transform(self, points, tf):
        """Apply 4x4 transform to point cloud"""
        homog = np.column_stack([points, np.ones(len(points))])
        return (tf @ homog.T).T[:,:3]