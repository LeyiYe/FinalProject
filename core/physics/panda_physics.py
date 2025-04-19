import numpy as np
from transforms3d.euler import euler2mat
from pysph.solver.application import Application
from pysph.sph.solid_mech.basic import ElasticSolidsScheme
from pysph.base.utils import get_particle_array
from core.utils.urdf_processor import URDFProcessor
from core.physics.panda_fk import get_fk
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PandaPhysics(Application):
    def __init__(self, urdf_file, particle_array):
        """Initialize physics simulation"""
        super().__init__()
        
        # Robot setup
        self.urdf_processor = URDFProcessor(urdf_file)
        self.hand_origin = np.eye(4)
        self.joint_positions = {}
        
        # Physics setup
        self.particle_array = self._prepare_object_particles(particle_array)
        self.boundaries = self._create_boundaries()
        
        # Configure scheme and solver parameters
        self.scheme = self._create_scheme()
        self.configure_solver()
        
        # Complete setup
        self.setup()

    def configure_solver(self):
        """Configure solver parameters"""
        from pysph.base.kernels import CubicSpline
        from pysph.sph.integrator import EPECIntegrator
        from pysph.sph.integrator_step import SolidMechStep
        
        self.kernel = CubicSpline(dim=3)
        self.integrator = EPECIntegrator(elastic=SolidMechStep())
        self.dt = 1e-5  # Time step
        self.tf = 1.0   # Final time

    def _prepare_object_particles(self, particle_array):
        """Add required properties for deformable object"""
        from pysph.sph.solid_mech.basic import get_particle_array_elastic_dynamics
        elastic_array = get_particle_array_elastic_dynamics(
            x=particle_array.x,
            y=particle_array.y,
            z=particle_array.z,
            m=particle_array.m,
            h=particle_array.h,
            rho=particle_array.rho,
            constants={
                'E': 1e6,    # Young's modulus (1 MPa)
                'nu': 0.45,  # Poisson's ratio
                'rho_ref': 1200  # Reference density
            }
        )
        elastic_array.add_property('tag', data=np.zeros(len(particle_array.x)))
        return elastic_array

    def _create_scheme(self):
        """Configure elastic solids scheme"""
        return ElasticSolidsScheme(
            elastic_solids=['object'],
            solids=['hand_boundary'],
            dim=3,
            artificial_stress_eps=0.3,
            alpha=1.0,
            beta=1.0
        )

    def create_particles(self):
        """Create all particle arrays"""
        return [self.particle_array] + self._create_boundary_particles()

    def _create_boundary_particles(self):
        """Generate boundary particles from URDF"""
        boundaries = []
        for name, points in self.boundaries.items():
            arr = get_particle_array(
                name=f'hand_boundary_{name}',
                x=points[:,0], y=points[:,1], z=points[:,2],
                m=np.ones(len(points))*0.1,
                h=np.ones(len(points))*0.005,
                tag=np.ones(len(points))
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

    def update_boundaries(self, joint_positions, hand_origin=None):
        """Update robot pose for new timestep"""
        self.joint_positions = joint_positions
        if hand_origin is not None:
            self.hand_origin = hand_origin
            
        # Update all boundary particles
        for name, points in self.boundaries.items():
            link_name = '_'.join(name.split('_')[:-2])
            tf = self._get_link_tf(link_name)
            self.boundaries[name] = self._apply_transform(points, tf)
            
            # Update solver particles
            if f'hand_boundary_{name}' in self.solver.particles.arrays:
                arr = self.solver.particles[f'hand_boundary_{name}']
                arr.x, arr.y, arr.z = points[:,0], points[:,1], points[:,2]

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

    def step(self):
        """Advance simulation by one timestep"""
        self.solver.step()

    def get_contacts(self):
        """Detect contacts between object and hand"""
        contacts = []
        obj = self.solver.particles.arrays[0]  # Deformable object
        
        for name in self.boundaries:
            boundary = self.solver.particles[f'hand_boundary_{name}']
            dists = np.sqrt(
                (obj.x[:,None] - boundary.x)**2 +
                (obj.y[:,None] - boundary.y)**2 +
                (obj.z[:,None] - boundary.z)**2
            )
            min_dists = np.min(dists, axis=1)
            
            for i, d in enumerate(min_dists):
                if d < self.scheme.h0:
                    contacts.append({
                        'particle_id': i,
                        'boundary': name,
                        'position': [obj.x[i], obj.y[i], obj.z[i]],
                        'force': max(0, 1e6 * (self.scheme.h0 - d))  # Linear repulsion
                    })
        return contacts

    def visualize(self):
        """Basic 3D visualization using matplotlib"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot object particles
        obj = self.solver.particles.arrays[0]
        ax.scatter(obj.x, obj.y, obj.z, c='blue', s=1, label='Object')
        
        # Plot hand particles
        for name in self.boundaries:
            if f'hand_boundary_{name}' in self.solver.particles.arrays:
                hand = self.solver.particles.arrays[f'hand_boundary_{name}']
                ax.scatter(hand.x, hand.y, hand.z, c='red', s=1, label='Hand')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.legend()
        plt.show()