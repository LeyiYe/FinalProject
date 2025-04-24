from enum import Enum, auto
import numpy as np
import pybullet as p
import time

class PandaState(Enum):
    INIT = auto()        # Initialization state
    APPROACH = auto()    # Move to pre-grasp position
    OPEN = auto()        # Open gripper fully
    CLOSE = auto()       # Close until contact detected
    GRASP = auto()       # Apply stabilizing force
    LIFT = auto()        # Raise the object
    HOLD = auto()        # Hold at target height
    DONE = auto()        # Final state

class PandaFSM:
    def __init__(self, panda_controller):
        self.state = PandaState.INIT
        self.controller = panda_controller
        self.timer = 0
        self.start_time = time.time()
        
        # Grasping parameters
        self.lift_height = 0.3  # meters
        self.contact_force_threshold = 5.0  # Newtons
        self.target_grasp_force = 15.0  # Newtons
        self.grasp_stable_counter = 0
        self.max_grasp_attempts = 3
        self.grasp_attempts = 0
        
        # Movement control
        self.arm_velocity = 0.05  # m/s
        self.gripper_velocity = 0.1  # m/s
        self.last_update_time = time.time()
        
        # State tracking
        self.initial_height = 0
        self.contact_force = np.zeros(3)
        self.grasp_force_history = []
        self.gripper_pos = [0, 0]
        
        # State name mapping
        self.state_names = {
            PandaState.INIT: "INIT",
            PandaState.APPROACH: "APPROACH",
            PandaState.OPEN: "OPEN",
            PandaState.CLOSE: "CLOSE",
            PandaState.GRASP: "GRASP",
            PandaState.LIFT: "LIFT",
            PandaState.HOLD: "HOLD",
            PandaState.DONE: "DONE"
        }

    def update(self):
        """Main FSM update loop"""
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        self.timer += 1

        # Print state every 0.1 seconds
        if self.timer % 10 == 0:
            print(f"[{current_time-self.start_time:.2f}s] State: {self.state_names[self.state]}")
        
        # Get current state information
        self.gripper_pos = self.controller.get_gripper_positions()
        gripper_center = self.get_gripper_center()
        self.contact_force = self.controller._compute_sph_reaction_force(gripper_center)
        
        # State machine logic
        if self.state == PandaState.INIT:
            self._init_state()
        elif self.state == PandaState.APPROACH:
            self._approach_state()
        elif self.state == PandaState.OPEN:
            self._open_state()
        elif self.state == PandaState.CLOSE:
            self._close_state()
        elif self.state == PandaState.GRASP:
            self._grasp_state()
        elif self.state == PandaState.LIFT:
            self._lift_state(dt)
        elif self.state == PandaState.HOLD:
            self._hold_state()
        elif self.state == PandaState.DONE:
            return False  # Stop simulation
        
        return True  # Continue running

    def _init_state(self):
        """Initialization state"""
        # Record initial height
        self.initial_height = self.get_gripper_center()[2]
        self.state = PandaState.APPROACH

    def _approach_state(self):
        """Move to pre-grasp position above object"""
        # Get object center
        obj_center = [
            np.mean(self.controller.sph_particles.x),
            np.mean(self.controller.sph_particles.y),
            np.max(self.controller.sph_particles.z)
        ]
        
        # Target position 5cm above object
        target_pos = [obj_center[0], obj_center[1], obj_center[2] + 0.05]
        
        # Move arm using IK
        self._move_arm_to_position(target_pos)
        
        # Check if we've reached the target
        current_pos = self.get_gripper_center()
        position_error = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
        
        if position_error < 0.01:  # 1cm tolerance
            print("Reached pre-grasp position")
            self.state = PandaState.OPEN

    def _open_state(self):
        """Fully open gripper before grasping"""
        open_position = 0.04  # Open position
        tolerance = 0.005
        
        # Check if already open enough
        if all(p > open_position - tolerance for p in self.gripper_pos):
            print("Gripper already open")
            self.state = PandaState.CLOSE
            return
            
        # Open gripper
        self.controller._open_gripper()
        
        # Check if fully open
        if all(p > open_position - tolerance for p in self.gripper_pos):
            print("Gripper fully open")
            self.state = PandaState.CLOSE

    def _close_state(self):
        """Close gripper until contact with object"""
        closing_speed = -0.05  # Slow closing speed
        force_magnitude = np.linalg.norm(self.contact_force)
        
        # Check if we've made contact
        if force_magnitude > self.contact_force_threshold:
            print(f"Contact detected (force: {force_magnitude:.2f}N)")
            self.state = PandaState.GRASP
            return
            
        # Close gripper gradually
        self.controller.set_gripper_velocity(closing_speed, closing_speed)
        
        # Safety check - don't close too far without detecting contact
        if all(p < 0.005 for p in self.gripper_pos):
            self.grasp_attempts += 1
            if self.grasp_attempts >= self.max_grasp_attempts:
                print("Failed to detect contact after multiple attempts")
                self.state = PandaState.DONE
            else:
                print("Retrying grasp attempt")
                self.state = PandaState.OPEN

    def _grasp_state(self):
        """Maintain stable grasp force"""
        current_force = np.linalg.norm(self.contact_force)
        force_error = self.target_grasp_force - current_force
        
        # Simple force controller
        torque = 0.1 + 0.05 * force_error  # Base torque + proportional term
        torque = np.clip(torque, 0.05, 0.5)  # Limit torque range
        
        # Apply symmetric torque to both fingers
        self.controller.set_gripper_torque(-torque, -torque)  # Negative = closing
        
        # Check stability (force maintained for 10 steps)
        if abs(force_error) < 2.0:
            self.grasp_stable_counter += 1
        else:
            self.grasp_stable_counter = 0
            
        if self.grasp_stable_counter > 10:
            print(f"Grasp stabilized at {current_force:.2f}N, beginning lift")
            self.state = PandaState.LIFT

    def _lift_state(self, dt):
        """Lift the object to target height"""
        current_pos = self.get_gripper_center()
        target_height = self.initial_height + self.lift_height
        
        # Check if we've reached the target height
        if current_pos[2] >= target_height - 0.01:
            print("Reached target height")
            self.state = PandaState.HOLD
            return
            
        # Calculate movement for this timestep
        distance_to_move = min(self.arm_velocity * dt, target_height - current_pos[2])
        target_pos = [current_pos[0], current_pos[1], current_pos[2] + distance_to_move]
        
        # Move arm upward
        self._move_arm_to_position(target_pos)
        
        # Monitor grasp force during lifting
        current_force = np.linalg.norm(self.contact_force)
        if current_force < self.target_grasp_force * 0.7:
            print("Warning: Grasp force dropping during lift!")
            # Could implement recovery behavior here

    def _hold_state(self):
        """Hold position at target height"""
        # Maintain current position and grasp force
        current_pos = self.get_gripper_center()
        self._move_arm_to_position(current_pos)
        
        # Maintain grasp force
        current_force = np.linalg.norm(self.contact_force)
        force_error = self.target_grasp_force - current_force
        torque = 0.1 + 0.05 * force_error
        self.controller.set_gripper_torque(-torque, -torque)
        
        # After holding for 2 seconds, finish
        if time.time() - self.last_update_time > 2.0:
            self.state = PandaState.DONE

    def _move_arm_to_position(self, target_pos):
        """Move arm to target position using IK"""
        target_orn = p.getQuaternionFromEuler([0, -math.pi, 0])  # Standard gripper orientation
        
        # Calculate IK solution
        joint_positions = p.calculateInverseKinematics(
            self.controller.panda,
            self.controller.end_effector_index,
            target_pos,
            targetOrientation=target_orn,
            lowerLimits=[-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
            upperLimits=[2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
            jointRanges=[5.8, 3.5, 5.8, 3.1, 5.8, 3.7, 5.8],
            restPoses=[0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32],
            maxNumIterations=100,
            residualThreshold=1e-5
        )
        
        # Set arm joints with velocity limits
        for i in range(7):  # First 7 joints are the arm
            p.setJointMotorControl2(
                self.controller.panda,
                i,
                p.POSITION_CONTROL,
                targetPosition=joint_positions[i],
                force=300,
                positionGain=0.3,
                maxVelocity=0.2  # Limit joint velocity
            )

    def get_gripper_center(self):
        """Compute midpoint between fingers"""
        left_pos = p.getLinkState(
            self.controller.panda,
            self.controller.joint_info['panda_finger_joint1']['index']
        )[0]
        right_pos = p.getLinkState(
            self.controller.panda,
            self.controller.joint_info['panda_finger_joint2']['index']
        )[0]
        return [
            (left_pos[0] + right_pos[0]) / 2,
            (left_pos[1] + right_pos[1]) / 2,
            (left_pos[2] + right_pos[2]) / 2
        ]