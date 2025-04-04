import isaacgym
from isaacgym import gymapi
import numpy as np

def main():
    # Initialize
    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z  # Important for Franka
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.use_gpu_pipeline = True  # Recommended for better performance
    
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    
    # Configure viewer
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        print("Failed to create viewer")
        return

    # Load Franka URDF - PATH MUST BE ABSOLUTE
    asset_root = "/home/ly1336/FinalProject/FinalProject/franka_description/robots/panda/"  # FULL PATH to package
    asset_file = "panda.urdf"  # Relative to asset_root
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.flip_visual_attachments = True

    asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX  # Helps with mesh loading
    asset_options.override_com = True  # Bypass some mesh issues
    asset_options.override_inertia = True
    asset_options.vhacd_enabled = True  # For collision meshes

    asset_options.disable_gravity = False
    asset_options.collapse_fixed_joints = False
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

    franka_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

    # Create env
    env_spacing = 2.0
    env = gym.create_env(sim, gymapi.Vec3(-env_spacing, -env_spacing, -env_spacing), 
                         gymapi.Vec3(env_spacing, env_spacing, env_spacing), 1)

    # Spawn Franka
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0, 0, 1.0)  # Z-up
    pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), np.pi)  # Rotate if needed

    franka_handle = gym.create_actor(env, franka_asset, pose, "franka", 0, 0)

    # Configure DOF properties
    dof_props = gym.get_actor_dof_properties(env, franka_handle)
    dof_props["driveMode"] = gymapi.DOF_MODE_POS
    dof_props["stiffness"] = np.array([1000.0] * 9)  # 7 arm joints + 2 fingers
    dof_props["damping"] = np.array([200.0] * 9)
    gym.set_actor_dof_properties(env, franka_handle, dof_props)

    num_dofs = gym.get_asset_dof_count(franka_asset)
    print(f"Asset has {num_dofs} DOFs")

    # Set initial joint positions
    dof_props = gym.get_actor_dof_properties(env, franka_handle)
    if num_dofs == 7:  # Only arm
        dof_props["stiffness"] = np.array([1000.0] * 7)
        dof_props["damping"] = np.array([200.0] * 7)
    elif num_dofs == 9:  # Arm + hand
        dof_props["stiffness"] = np.array([1000.0] * 9)
        dof_props["damping"] = np.array([200.0] * 9)


    gym.set_actor_dof_properties(env, franka_handle, dof_props)

    dof_state = np.zeros(gym.get_actor_dof_count(env, franka_handle), dtype=gymapi.DofState.dtype)
    dof_state["pos"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.04]  # 7 arm joints + 2 fingers
    gym.set_actor_dof_states(env, franka_handle, dof_state, gymapi.STATE_POS)
    

    # Camera setup
    cam_pos = gymapi.Vec3(2, 2, 2)
    cam_target = gymapi.Vec3(0, 0, 1)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    # Simulation loop
    while not gym.query_viewer_has_closed(viewer):
        # Step physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        
        # Update viewer
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

if __name__ == "__main__":
    main()