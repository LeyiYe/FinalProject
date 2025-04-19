import numpy as np
import pysph
from scipy.linalg import expm
from pysph.sph.solver.solver import Solver
from pysph.base.utils import utils
from transforms3d.euler import euler2mat
from core.robot.urdf_processor import URDFProcessor
from panda_fk import get_fk, wedge, skew

class PandaPhysics:
    def __init__(self, urdf_file, particle_array):
        """
        Initialize physics simulation with Panda hand and deformable object
        
        Args:
            urdf_file (str): Path to Panda hand URDF file
            particle_array: PySPH particle array for the deformable object
        """
        self.urdf_processor = URDFProcessor(urdf_file)
        self.boundaries = self._create_boundaries()
        self.solver = self._setup_solver(particle_array)
        self.hand_origin = np.eye(4)  # Will be updated with actual hand origin
        self.joint_positions = {}  # Dictionary to store joint states
        
    def _create_boundaries(self):
        """Create boundary particles from URDF collision geometries"""
        boundaries = {}
        
        # Get collision shapes from URDF
        for link_name, link_data in self.urdf_processor.links.items():
            for i, collision in enumerate(link_data['collisions']):
                boundary_name = f"{link_name}_collision_{i}"
                boundaries[boundary_name] = self._create_boundary_from_collision(collision)
                
        return boundaries
    
    def _create_boundary_from_collision(self, collision):
        """Create boundary particles for a specific collision geometry"""
        geo_type = collision['type']
        params = collision['params']
        pos = collision['position']
        rot = collision['rotation']
        
        # Create points based on geometry type
        if geo_type == 'box':
            size = [float(x) for x in params['size'].split()]
            x, y, z = np.meshgrid(
                np.linspace(-size[0]/2, size[0]/2, 5),
                np.linspace(-size[1]/2, size[1]/2, 5),
                np.linspace(-size[2]/2, size[2]/2, 5)
            )
            points = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
            
        elif geo_type == 'sphere':
            radius = float(params['radius'])
            # Create points on sphere surface
            theta, phi = np.mgrid[0:2*np.pi:10j, 0:np.pi:10j]
            x = radius * np.cos(theta) * np.sin(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(phi)
            points = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
            
        elif geo_type == 'cylinder':
            radius = float(params['radius'])
            length = float(params['length'])
            # Create points on cylinder surface
            theta, h = np.mgrid[0:2*np.pi:10j, -length/2:length/2:5j]
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            z = h
            points = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
            
        else:  # For mesh or unsupported types, create simple placeholder
            points = np.array([[0, 0, 0]])
            
        # Apply collision geometry transform
        homog_points = np.hstack([points, np.ones((len(points), 1))])
        transform = np.eye(4)
        transform[:3, :3] = rot
        transform[:3, 3] = pos
        transformed_points = (transform @ homog_points.T).T[:, :3]
        
        return transformed_points
    
    def _setup_solver(self, particle_array):
        """Configure PySPH solver with particles and boundaries"""
        solver = Solver()
        solver.add_particles(particle_array)
        
        for name, boundary in self.boundaries.items():
            # Create boundary particle array
            boundary_particles = utils.get_particle_array(
                name=name,
                x=boundary[:,0],
                y=boundary[:,1],
                z=boundary[:,2],
                m=np.ones(len(boundary))*0.001,
                h=np.ones(len(boundary))*0.005
            )
            solver.add_boundary(boundary_particles)
            
        return solver
    
    def _get_link_fk(self, link_name):
        """
        Get transform for a specific link using current joint positions
        
        Args:
            link_name: Name of the link to get transform for
            
        Returns:
            4x4 transformation matrix for the link
        """
        # Convert joint positions dictionary to array in correct order
        # Note: You'll need to ensure joint order matches your FK implementation
        joint_array = np.zeros(16)  # Adjust size based on your FK
        for i in range(16):  # Assuming 16 joints as in your FK
            joint_array[i] = self.joint_positions.get(f'joint_{i}', 0.0)
        
        # Get FK transform for this link
        if 'left' in link_name.lower():
            mode = 'left'
        elif 'right' in link_name.lower():
            mode = 'right'
        else:
            mode = 'mid'
            
        # Create hand origin transform from current state
        hand_origin = self.hand_origin  # Should be updated via update_boundaries
        
        # Get the full transform using your FK function
        transform = get_fk(joint_array, hand_origin, mode=mode)
        
        return transform
    
    def update_boundaries(self, joint_positions, hand_origin=None):
        """
        Update boundary positions based on joint states and hand origin
        
        Args:
            joint_positions: Dictionary of joint_name:position values
            hand_origin: Optional 4x4 transform matrix for hand base
        """
        # Update stored joint positions
        self.joint_positions = joint_positions
        
        if hand_origin is not None:
            self.hand_origin = hand_origin
            
        # Update all boundaries based on their link transforms
        for boundary_name in self.boundaries:
            # Extract link name from boundary name (format: linkname_collision_X)
            link_name = '_'.join(boundary_name.split('_')[:-2])
            link_tf = self._get_link_fk(link_name)
            self._apply_transform_to_boundary(boundary_name, link_tf)
        
        # Update solver with new boundary positions
        for name in self.boundaries:
            self.solver.boundary_arrays[name].x = self.boundaries[name][:,0]
            self.solver.boundary_arrays[name].y = self.boundaries[name][:,1]
            self.solver.boundary_arrays[name].z = self.boundaries[name][:,2]
    
    def _apply_transform_to_boundary(self, name, transform):
        """Apply 4x4 transform to boundary particles"""
        # Convert to homogeneous coordinates
        ones = np.ones((len(self.boundaries[name]), 1))
        homog = np.hstack([self.boundaries[name], ones])
        
        # Apply transform
        transformed = (transform @ homog.T).T
        
        # Update boundary positions (ignore homogeneous coordinate)
        self.boundaries[name] = transformed[:,:3]
    
    def step(self):
        """Advance physics simulation by one timestep"""
        self.solver.step()
    
    def get_contacts(self):
        """
        Detect contacts between object and hand
        
        Returns:
            List of contact dictionaries with:
            - particle_id: ID of contacting particle
            - boundary: Name of boundary element
            - position: Contact position
            - force: Contact force magnitude
        """
        contacts = []
        obj_particles = self.solver.particles[0]  # Assuming object is first
        
        for boundary_name in self.boundaries:
            boundary_particles = self.solver.boundary_arrays[boundary_name]
            
            # Simple contact detection - in practice use spatial hashing
            for i, (px, py, pz) in enumerate(zip(obj_particles.x, 
                                                obj_particles.y, 
                                                obj_particles.z)):
                # Find nearest boundary particle
                dists = np.sqrt(
                    (px - boundary_particles.x)**2 +
                    (py - boundary_particles.y)**2 +
                    (pz - boundary_particles.z)**2
                )
                min_dist = np.min(dists)
                
                if min_dist < self.solver.kernel.radius:
                    contacts.append({
                        'particle_id': i,
                        'boundary': boundary_name,
                        'position': [px, py, pz],
                        'force': self._calculate_repulsion_force(min_dist)
                    })
        
        return contacts
    
    def _calculate_repulsion_force(self, distance):
        """Simple force model based on penetration distance"""
        kernel_radius = self.solver.kernel.radius
        if distance >= kernel_radius:
            return 0.0
        return (kernel_radius - distance) * 1000  # Arbitrary stiffness