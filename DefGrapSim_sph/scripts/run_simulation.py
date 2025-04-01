import hydra
import numpy as np
from omegaconf import DictConfig
from core.sph_simulator import SPHSimulator
from core.isaac_interface import IsaacVisualizer
from objects.factory import ObjectFactory

@hydra.main(config_path="../configs", config_name="default")
def run_simulation(cfg: DictConfig):
    # Initialize SPH simulation with configuration
    sph_sim = SPHSimulator(cfg.sph)
    
    # Create deformable object through factory
    obj = ObjectFactory.create_object(cfg.object.type, cfg.object)
    sph_sim.setup_simulation([obj])
    
    # Initialize IsaacGym
    gym, sim, env = initialize_isaac(cfg.isaac)
    
    # Prepare visualization config (combine object and visualization params)
    vis_config = {
        **cfg.object,  # Includes radius, spacing etc.
        **cfg.visualization,  # Particle display size, optimization settings
        'estimated_particles': len(obj.get_particle_array().x)  # Actual particle count
    }
    
    # Initialize visualizer
    visualizer = IsaacVisualizer(gym, sim, env)
    visualizer.create_visual_assets(cfg.object.type, vis_config)
    
    # Initial settling period (optional)
    for _ in range(cfg.simulation.settle_steps):
        sph_sim.step(cfg.simulation.dt)
    
    # Main simulation loop
    frame_count = 0
    while True:
        # Simulation step
        sph_sim.step(cfg.simulation.dt)
        
        # Get particle state (format depends on your SPH implementation)
        particle_state = sph_sim.get_particle_state()
        
        # Update visualization (throttle if needed)
        if frame_count % cfg.visualization.update_interval == 0:
            visualizer.update_visualization(particle_state)
            
        # Optional: Performance logging
        if frame_count % 100 == 0:
            log_performance(cfg, frame_count)
            
        frame_count += 1
        
        # Exit conditions
        if cfg.simulation.max_steps and frame_count >= cfg.simulation.max_steps:
            break

def initialize_isaac(config):
    """Initialize Isaac Gym environment"""
    gym = gymapi.acquire_gym()
    
    # Configure simulator
    sim_params = gymapi.SimParams()
    sim_params.dt = config.dt
    sim_params.substeps = config.substeps
    sim_params.gravity = gymapi.Vec3(*config.gravity)
    
    # Graphics/Physics backend
    sim_params.physx.use_gpu = config.use_gpu
    sim_params.physx.num_threads = config.num_threads
    
    # Create simulation
    sim = gym.create_sim(
        config.compute_device_id,
        config.graphics_device_id,
        config.physics_engine,
        sim_params
    )
    
    # Create default environment
    env = gym.create_env(sim, 
                        gymapi.Vec3(*config.env_spacing),
                        config.num_envs)
    
    # Add ground plane
    plane_params = gymapi.PlaneParams()
    gym.add_ground(sim, plane_params)
    
    return gym, sim, env

def log_performance(config, frame_count):
    """Log simulation performance metrics"""
    # Implement your performance logging here
    print(f"Step {frame_count}: {1/(time.time()-last_time):.1f} FPS")

if __name__ == "__main__":
    run_simulation()