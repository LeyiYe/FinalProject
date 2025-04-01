import numpy as np
from isaacgym import gymapi

class IsaacVisualizer:
    def __init__(self, gym, sim, env):
        self.gym = gym
        self.sim = sim
        self.env = env
        self.visual_assets = {}
        self.actor_handles = []
        self.particle_radius = None  # Track particle visualization size
        
    def create_visual_assets(self, object_type, properties):
        """Create visual representation for an object type"""
        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = True
        asset_options.fix_base_link = True
        
        # Store particle radius for later use
        self.particle_radius = properties.get('particle_radius', 
                                           properties.get('spacing', 0.1) * 0.5)
        
        # Create different assets based on object type
        if object_type == 'sphere':
            # Create a simple sphere asset for each particle
            asset = self.gym.create_sphere(
                self.sim, 
                self.particle_radius,
                asset_options
            )
            self.visual_assets[object_type] = asset
            
            # Create multiple instances for all particles
            self._create_particle_instances(properties)
            
    def _create_particle_instances(self, properties):
        """Create actor instances for all particles"""
        # Clear existing handles if any
        self.actor_handles = []
        
        # Create initial transforms (positions will be updated)
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, 0, 0)  # Temporary position
        
        # Create one actor per particle
        num_particles = properties.get('estimated_particles', 1000)
        for i in range(num_particles):
            actor_handle = self.gym.create_actor(
                self.env, 
                self.visual_assets['sphere'], 
                pose,
                f"particle_{i}", 
                0,  # collision group
                0   # collision filter
            )
            self.actor_handles.append(actor_handle)
            
    def update_visualization(self, particle_state):
        """Update visualization based on current particle state"""
        if not self.actor_handles:
            return
            
        # Convert particle state to numpy array if needed
        if isinstance(particle_state, dict):
            positions = np.column_stack([
                particle_state['x'],
                particle_state['y'],
                particle_state['z']
            ])
        else:
            positions = np.array(particle_state)
        
        # Update all particle positions
        for i, pos in enumerate(positions[:len(self.actor_handles)]):
            props = self.gym.get_actor_rigid_shape_properties(
                self.env, 
                self.actor_handles[i]
            )
            props[0].position = gymapi.Vec3(*pos)
            self.gym.set_actor_rigid_shape_properties(
                self.env, 
                self.actor_handles[i], 
                props
            )
            
    def optimize_visualization(self, method='point_cloud'):
        """Optimize visualization performance"""
        if method == 'point_cloud':
            # Alternative approach using fewer visual elements
            self._setup_point_cloud()
            
    def _setup_point_cloud(self):
        """Experimental: Use point cloud for better performance"""
        # This would require custom rendering setup
        pass