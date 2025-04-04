import isaacgym
from isaacgym import gymapi

# Initialize
gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim = gym.create_sim(..., sim_params)

# Load Franka URDF
asset_root = "/franka_ros/franka_description/robots/panda"
asset_file = "panda.urdf"  # or panda_with_hand.urdf
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.flip_visual_attachments = True  # Fix hand orientation if needed

franka_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)


# Create env
env = gym.create_env(sim, gymapi.Vec3(-2, 0, -2), gymapi.Vec3(2, 2, 2), 1)
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0, 0, 1.0)  # Spawn above ground

franka_handle = gym.create_actor(env, franka_asset, pose, "franka", 0, 0)

# Set PD control for fingers
dof_props = gym.get_actor_dof_properties(env, franka_handle)
dof_props["driveMode"] = gymapi.DOF_MODE_POS
dof_props["stiffness"] = 1000.0
dof_props["damping"] = 200.0
gym.set_actor_dof_properties(env, franka_handle, dof_props)

# Close fingers (0.04 = fully closed)
gym.set_dof_target_position(env, franka_handle, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.04])

# Simulation loop
while True:
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
