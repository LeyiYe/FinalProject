import isaacgym
from isaacgym import gymapi
import numpy as np

def main():
    # Initialize
    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.use_gpu_pipeline = False
    sim_params.physx.num_threads = 4
    sim_params.physx.use_gpu = False

    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    
    # Configure viewer
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        print("Failed to create viewer")
        return

    # Load HAND-ONLY URDF
    asset_root = "/home/ly1336/FinalProject/FinalProject/franka_description/robots/common/"
    asset_file = "hand.urdf"  # Using hand-only URDF
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True  # Critical for hand-only
    asset_options.disable_gravity = True  # Better for gripper control
    asset_options.flip_visual_attachments = True
    asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
    asset_options.vhacd_enabled = True

    hand_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

    # Create env
    env_spacing = 1.0
    env = gym.create_env(sim, gymapi.Vec3(-env_spacing, -env_spacing, -env_spacing), 
                         gymapi.Vec3(env_spacing, env_spacing, env_spacing), 1)

    # Spawn Hand
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0, 0, 1.5)  # Higher for better visibility
    hand_handle = gym.create_actor(env, hand_asset, pose, "hand", 0, 0)

    # Configure ONLY FINGER JOINTS (2 DOFs)
    dof_props = gym.get_actor_dof_properties(env, hand_handle)
    print(f"Hand has {len(dof_props['stiffness'])} DOFs")  # Should be 2
    
    dof_props["driveMode"] = gymapi.DOF_MODE_POS
    dof_props["stiffness"] = [1000.0, 1000.0]  # Only 2 values needed
    dof_props["damping"] = [200.0, 200.0]
    gym.set_actor_dof_properties(env, hand_handle, dof_props)

    # Set initial finger positions (0=open, 0.04=closed)
    gym.set_actor_dof_position_targets(env, hand_handle, [0.02, 0.02])  # Half-closed

    # Camera setup focused on gripper
    cam_pos = gymapi.Vec3(0.5, 0.5, 1.5)  # Closer view
    cam_target = gymapi.Vec3(0, 0, 1.3)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    # Simple grasping sequence
    step = 0
    while not gym.query_viewer_has_closed(viewer):
        # Animate fingers
        if step < 30:
            # Close gradually
            pos = 0.02 + (0.02 * step/30)
            gym.set_actor_dof_position_targets(env, hand_handle, [pos, pos])
        elif 60 < step < 90:
            # Open gradually
            pos = 0.04 - (0.02 * (step-60)/30)
            gym.set_actor_dof_position_targets(env, hand_handle, [pos, pos])
        
        # Step simulation
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)
        step += 1

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

if __name__ == "__main__":
    main()