# isaacgym_visualizer.py
import isaacgym
from isaacgym import gymapi
import numpy as np
from object.pysph_simulation import DeformableObjectSimulation

class IsaacGymVisualizer:
    def __init__(self):
        self.gym = gymapi.acquire_gym()
        self.sim = None
        self.viewer = None
        self.env = None
        self.hand_handle = None
        self.particle_actors = []
        self.sph_sim = DeformableObjectSimulation()
        
    def setup_simulation(self):
        """Initialize IsaacGym simulation with FleX"""
        # FleX simulation parameters
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        
        # FleX-specific parameters
        sim_params.flex.solver_type = 5  # GPU solver
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 20
        sim_params.flex.relaxation = 0.8
        sim_params.flex.warm_start = 0.5
        
        # Enable GPU pipeline for FleX
        sim_params.use_gpu_pipeline = True
        
        # Create FleX simulation
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_FLEX, sim_params)
        
        # Create viewer
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            raise RuntimeError("Failed to create viewer")
        
        # Create environment
        env_spacing = 1.0
        self.env = self.gym.create_env(
            self.sim,
            gymapi.Vec3(-env_spacing, -env_spacing, -env_spacing),
            gymapi.Vec3(env_spacing, env_spacing, env_spacing),
            1
        )
        
        # Load hand asset with FleX settings
        self._load_hand()
        
        # Setup particles with FleX properties
        self._setup_particles()
        
        # Setup camera
        self._setup_camera()
    
    def _load_hand(self):
        """Load the Panda hand asset with FleX settings"""
        asset_root = "/home/ly1336/FinalProject/FinalProject/franka_description/robots/common/"
        asset_file = "hand.urdf"
        
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = True
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        
        # FleX-specific hand settings
        asset_options.flex.disable_gravity = True
        asset_options.flex.shape_collision_margin = 0.01
        asset_options.flex.dynamic_friction = 0.5
        asset_options.flex.static_friction = 0.5

        hand_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # Spawn Hand
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, 0, 1.5)
        pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), np.pi)
        self.hand_handle = self.gym.create_actor(self.env, hand_asset, pose, "hand", 0, 0)

        # Configure finger joints
        dof_props = self.gym.get_actor_dof_properties(self.env, self.hand_handle)
        dof_props["driveMode"] = gymapi.DOF_MODE_POS
        dof_props["stiffness"] = [1000.0, 1000.0]
        dof_props["damping"] = [200.0, 200.0]
        self.gym.set_actor_dof_properties(self.env, self.hand_handle, dof_props)
        
        # Set initial finger positions
        self.gym.set_actor_dof_position_targets(self.env, self.hand_handle, [0.02, 0.02])
    
    def _setup_particles(self):
        """Create FleX particle representation"""
        particle_radius = 0.02
        
        # FleX particle asset options
        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = True
        asset_options.flex.disable_gravity = True
        asset_options.flex.collision_distance = particle_radius * 0.5
        asset_options.flex.particle_friction = 0.1
        asset_options.flex.particle_damping = 0.01
        asset_options.flex.particle_adhesion = 0.0
        
        sphere_asset = self.gym.create_sphere(self.sim, particle_radius, asset_options)
        
        initial_state = self.sph_sim.get_initial_state()
        
        for i in range(len(initial_state['x'])):
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(
                initial_state['x'][i],
                initial_state['y'][i],
                initial_state['z'][i] + 1.0
            )
            
            actor = self.gym.create_actor(
                self.env, 
                sphere_asset, 
                pose, 
                f"particle_{i}", 
                0, 0
            )
            
            # Set FleX particle properties if available
            if hasattr(self.gym, 'get_actor_flex_particle_properties'):
                props = self.gym.get_actor_flex_particle_properties(self.env, actor)
                props['mass'] = 0.1  # Adjust mass as needed
                props['radius'] = particle_radius
                self.gym.set_actor_flex_particle_properties(self.env, actor, props)
            
            # Color setting (if available)
            if hasattr(self.gym, 'set_rigid_body_color'):
                self.gym.set_rigid_body_color(self.env, actor, 0, gymapi.MeshType(0), gymapi.Vec3(0.2, 0.6, 1.0))
            
            self.particle_actors.append(actor)
    
    def _setup_camera(self):
        """Configure the camera view"""
        cam_pos = gymapi.Vec3(0.5, 0.5, 1.5)
        cam_target = gymapi.Vec3(0, 0, 1.3)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
    
    def update_particles(self):
        """Update particle positions from SPH simulation"""
        state = self.sph_sim.step()
        for i, actor in enumerate(self.particle_actors):
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(
                state['x'][i],
                state['y'][i],
                state['z'][i] + 1.0  # Same Z offset as initial setup
            )
            self.gym.set_actor_transform(self.env, actor, pose)
    
    def run_simulation(self):
        """Main simulation loop"""
        step = 0
        while not self.gym.query_viewer_has_closed(self.viewer):
            # Update particle positions
            self.update_particles()
            
            # Animate fingers
            if step < 30:
                pos = 0.02 + (0.02 * step/30)
                self.gym.set_actor_dof_position_targets(self.env, self.hand_handle, [pos, pos])
            elif 60 < step < 90:
                pos = 0.04 - (0.02 * (step-60)/30)
                self.gym.set_actor_dof_position_targets(self.env, self.hand_handle, [pos, pos])
            
            # Step simulation
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)
            step += 1
        
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

if __name__ == "__main__":
    visualizer = IsaacGymVisualizer()
    visualizer.setup_simulation()
    visualizer.run_simulation()