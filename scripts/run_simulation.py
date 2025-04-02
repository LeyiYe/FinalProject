import hydra
import time

from FinalProject.core.sph_simulator import SPHSimulator
from FinalProject.core.isaac_interface import IsaacVisualizer
from FinalProject.objects.factory import ObjectFactory

from omegaconf import DictConfig
from isaacgym import gymapi
#from core.sph_simulator import SPHSimulator
#from core.isaac_interface import IsaacVisualizer
#from objects.factory import ObjectFactory

@hydra.main(config_path="../configs", config_name="default")
def run_simulation(cfg: DictConfig):
    # Track performance
    last_time = time.time()
    
    # Initialize SPH simulation
    sph_sim = SPHSimulator(cfg.sph)
    obj = ObjectFactory.create_object(cfg.object.type, cfg.object)
    sph_sim.setup_simulation([obj])
    
    # Initialize IsaacGym with viewer
    gym, sim, env, viewer = initialize_isaac(cfg.isaac)
    
    # Setup visualization
    vis_config = {
        **cfg.object,
        **cfg.visualization,
        'estimated_particles': len(obj.get_particle_array().x)
    }
    visualizer = IsaacVisualizer(gym, sim, env)
    visualizer.create_visual_assets(cfg.object.type, vis_config)

    # Camera setup
    cam_props = gymapi.CameraProperties()
    cam_props.horizontal_fov = 75.0
    cam_pos = gymapi.Vec3(0, -1.5, 0.5)
    cam_target = gymapi.Vec3(0, 0, 0.2)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    # Main simulation loop
    frame_count = 0
    try:
        while not gym.query_viewer_has_closed(viewer):
            # Simulation step
            sph_sim.step(cfg.simulation.dt)
            
            # Visualization update
            if frame_count % cfg.visualization.update_interval == 0:
                particle_state = sph_sim.get_particle_state()
                visualizer.update_visualization(particle_state)
                
                # Render frame
                gym.step_graphics(sim)
                gym.draw_viewer(viewer, sim, True)
                gym.sync_frame_time(sim)

            # Performance logging
            if frame_count % 100 == 0:
                current_time = time.time()
                fps = 100/(current_time - last_time)
                print(f"Step {frame_count}: {fps:.1f} FPS")
                last_time = current_time
                
            frame_count += 1
            
            # Exit conditions
            if cfg.simulation.max_steps and frame_count >= cfg.simulation.max_steps:
                break

    except KeyboardInterrupt:
        print("Simulation stopped by user")
    finally:
        gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)

def initialize_isaac(config):
    """Initialize Isaac Gym environment with viewer"""
    gym = gymapi.acquire_gym()
    
    # Configure simulator
    sim_params = gymapi.SimParams()
    sim_params.dt = config.dt
    sim_params.substeps = config.substeps
    sim_params.gravity = gymapi.Vec3(*config.gravity)
    sim_params.physx.use_gpu = config.use_gpu
    sim_params.physx.num_threads = config.num_threads
    
    # Create simulation
    sim = gym.create_sim(
        config.compute_device_id,
        config.graphics_device_id,
        gymapi.SIM_PHYSX,
        sim_params
    )
    
    # Create environment
    env = gym.create_env(sim, gymapi.Vec3(*config.env_spacing), config.num_envs)
    
    # Add ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    plane_params.static_friction = 0.5
    plane_params.dynamic_friction = 0.5
    gym.add_ground(sim, plane_params)
    
    # Create viewer
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise RuntimeError("Failed to create viewer")
    
    return gym, sim, env, viewer

if __name__ == "__main__":
    run_simulation()