# Copyright (c) 2020 NVIDIA Corporation
# Modified to support arm-only simulation

"""Simplified FSM for Franka Panda arm control without object interaction."""

import numpy as np
from isaacgym import gymapi
from utils import panda_fk

DEBUG = False

class PandaFsm:
    """FSM for control of Panda hand without object interaction."""

    def __init__(self, cfg, gym_handle, sim_handle, env_handles, franka_handle,
                 grasp_transform, env_id, hand_origin, viewer, env_dim, mode):
        """Initialize simplified FSM for arm control."""
        self.mode = mode
        self.started = False
        self.state = 'open'
        self.cfg = cfg

        # Simulation handles
        self.gym_handle = gym_handle
        self.sim_handle = sim_handle
        self.env_handles = env_handles
        self.env_id = env_id
        self.env_handle = self.env_handles[self.env_id]
        self.viewer = viewer

        # Sim params
        self.sim_params = gymapi.SimParams()
        self.sim_params = self.gym_handle.get_sim_params(self.sim_handle)
        #self.envs_per_row = envs_per_row
        self.env_dim = env_dim
        self.env_x_offset = 2. * self.env_dim * (self.env_id % self.envs_per_row)
        self.env_z_offset = 2. * self.env_dim * int(self.env_id / self.envs_per_row)

        # Franka actor
        self.franka_handle = franka_handle
        num_franka_bodies = self.gym_handle.get_actor_rigid_body_count(
            self.env_handle, self.franka_handle)
        self.finger_indices = [num_franka_bodies - 2, num_franka_bodies - 1]
        self.hand_indices = range(num_franka_bodies)
        self.left_finger_handle = self.gym_handle.get_actor_rigid_body_handle(
            self.env_handle, self.franka_handle, self.finger_indices[-2])
        self.right_finger_handle = self.gym_handle.get_actor_rigid_body_handle(
            self.env_handle, self.franka_handle, self.finger_indices[-1])

        # Franka control
        self.grasp_transform = grasp_transform
        self.franka_dof_states = None
        self.hand_origin = hand_origin
        
        # Finger positions
        self.mid_finger_position = np.array([
            self.hand_origin.p.x,
            self.hand_origin.p.y,
            self.hand_origin.p.z + self.cfg['franka']['gripper_tip_z_offset'],
            1
        ])
        
        # Control outputs
        self.vel_des = np.zeros(self.cfg['franka']['num_joints'])
        self.pos_des = np.zeros(self.cfg['franka']['num_joints'])
        self.torque_des = np.zeros(self.cfg['franka']['num_joints'])
        self.running_torque = [-0.1, -0.1]

        # State tracking
        self.close_fails = 0
        self.left_has_contacted = False
        self.right_has_contacted = False
        self.full_counter = 0
        self.timed_out = False

    def lock_maximum_finger_positions(self, tolerance):
        """Set upper gripper limit as current positions plus tolerance."""
        dof_props = self.gym_handle.get_actor_dof_properties(
            self.env_handle, self.franka_handle)
        curr_pos = self.franka_dof_states['pos'][-2:]
        dof_props['upper'][-2:] = curr_pos + tolerance
        self.gym_handle.set_actor_dof_properties(
            self.env_handle, self.franka_handle, dof_props)

    def lock_minimum_finger_positions(self, tolerance):
        """Set lower gripper limit as current positions minus tolerance."""
        dof_props = self.gym_handle.get_actor_dof_properties(
            self.env_handle, self.franka_handle)
        curr_pos = self.franka_dof_states['pos'][-2:]
        dof_props['lower'][-2:] = curr_pos - tolerance
        self.gym_handle.set_actor_dof_properties(
            self.env_handle, self.franka_handle, dof_props)

    def save_full_state(self):
        """Save current arm state."""
        self.saved_franka_state = np.copy(
            self.gym_handle.get_actor_dof_states(
                self.env_handle, self.franka_handle, gymapi.STATE_ALL))
        self.saved_fsm_state = self.state

    def reset_saved_state(self):
        """Revert to previously saved arm state."""
        self.gym_handle.set_actor_dof_states(
            self.env_handle,
            self.franka_handle,
            self.saved_franka_state,
            gymapi.STATE_ALL)
        self.state = self.saved_fsm_state

    def run_state_machine(self):
        """Simplified state machine for arm control."""
        self.full_counter += 1
        self.franka_dof_states = self.gym_handle.get_actor_dof_states(
            self.env_handle, self.franka_handle, gymapi.STATE_ALL)

        ################################################################
        # OPEN STATE: Initialize the hand with open fingers
        ################################################################
        if self.state == 'open':
            self.started = True
            self.state = "close"  # Immediately transition to closing
            self.save_full_state()

        ################################################################
        # CLOSE STATE: Close fingers
        ################################################################
        elif self.state == 'close':
            closing_speeds = np.zeros(self.cfg['franka']['num_joints'])
            closing_speeds[-2:] = -0.7 * np.ones(2)  # Close fingers
            self.vel_des = np.copy(closing_speeds)

            # Check if fingers are fully closed
            if np.all(self.franka_dof_states['pos'][-2:] < 0.001):
                self.state = 'done'

        ################################################################
        # DONE STATE: Stop all movement
        ################################################################
        elif self.state == 'done':
            self.vel_des = np.zeros(self.cfg['franka']['num_joints'])

        # Apply desired velocities
        self.vel_des = np.asarray(self.vel_des, dtype=np.float32)
        self.gym_handle.set_actor_dof_velocity_targets(
            self.env_handle, self.franka_handle, self.vel_des)

        # Set joint damping
        dof_props = self.gym_handle.get_actor_dof_properties(
            self.env_handle, self.franka_handle)
        dof_props['damping'][-2:] = np.repeat(self.cfg['franka']['joint_damping'], 2)
        self.gym_handle.set_actor_dof_properties(
            self.env_handle, self.franka_handle, dof_props)