from enum import Enum, auto
import numpy as np
import pybullet as p

class PandaState(Enum):
    OPEN = auto()       # Open gripper fully
    APPROACH = auto()   # Move to pre-grasp position
    CLOSE = auto()      # Close until contact detected
    GRASP = auto()      # Apply stabilizing force
    LIFT = auto()       # Raise the object
    DONE = auto()       # Final state

class PandaFSM:
    def __init__(self, panda_controller):
        self.state = PandaState.OPEN
        self.controller = panda_controller
        self.timer = 0
        self._init_variables()

    def _init_variables(self):
        """Initialize state variables"""
        self.contact_force = np.zeros(3)  # Track SPH reaction forces
        self.grasp_force_history = []
        self.lift_height = 0.3  # meters
        self.grasp_stable_counter = 0

    def update(self):
        """Main FSM update loop"""
        self.timer += 1
        
        # Get current gripper state
        gripper_center = self.get_gripper_center()
        gripper_pos = self.controller.get_gripper_positions()
        
        # Compute SPH reaction force for contact detection
        self.contact_force = self.controller._compute_sph_reaction_force(gripper_center)
        
        # State machine logic
        if self.state == PandaState.OPEN:
            self._open_state()
        elif self.state == PandaState.APPROACH:
            self._approach_state()
        elif self.state == PandaState.CLOSE:
            self._close_state()
        elif self.state == PandaState.GRASP:
            self._grasp_state()
        elif self.state == PandaState.LIFT:
            self._lift_state()
        elif self.state == PandaState.DONE:
            self.controller.set_gripper_velocity(0, 0)
            self.controller.set_gripper_torque(0, 0)
            return False  # Stop simulation
        
        return True  # Continue running

    def _open_state(self):
        """Fully open gripper before approach"""
        self.controller.set_gripper_velocity(0.5, 0.5)  # Open fast
        
        # Transition when fully open (joint positions > threshold)
        if all(p > 0.04 for p in self.controller.get_gripper_positions()):
            print("Gripper fully open, approaching object")
            self.state = PandaState.APPROACH

    def _approach_state(self):
        """Position gripper above the object"""
        target_pos = [0.5, -0.5, 0.15]  # Above platform center
        current_pos = self.get_gripper_center()
        
        # Calculate IK for target position
        target_joints = p.calculateInverseKinematics(
            self.controller.panda,
            self.controller.joint_info['panda_hand_joint']['index'],
            target_pos,
            targetOrientation=p.getQuaternionFromEuler([0, -np.pi, 0]))
        
        # Move arm joints (first 7)
        for i in range(7):
            p.setJointMotorControl2(
                self.controller.panda,
                i,
                p.POSITION_CONTROL,
                targetPosition=target_joints[i],
                force=100)
        
        # Check if reached position
        if np.linalg.norm(np.array(target_pos) - np.array(current_pos)) < 0.02:
            print("Approach complete, starting to close")
            self.state = PandaState.CLOSE

    def _close_state(self):
        """Close gripper until contact with deformable object"""
        closing_speed = -0.3  # Slow closing speed
        self.controller.set_gripper_velocity(closing_speed, closing_speed)
        
        # Check SPH reaction force magnitude for contact
        contact_threshold = 5.0  # Newtons
        force_magnitude = np.linalg.norm(self.contact_force)
        
        if force_magnitude > contact_threshold:
            print(f"Contact detected (force: {force_magnitude:.2f}N), stabilizing grasp")
            self.state = PandaState.GRASP
            self.initial_grasp_force = force_magnitude

    def _grasp_state(self):
        """Maintain stable grasp force before lifting"""
        target_force = 15.0  # Newtons
        current_force = np.linalg.norm(self.contact_force)
        
        # Simple force controller
        force_error = target_force - current_force
        torque = 0.1 + 0.05 * force_error  # Base torque + proportional term
        
        # Apply symmetric torque to both fingers
        self.controller.set_gripper_torque(-torque, -torque)  # Negative = closing
        
        # Check stability (force maintained for 10 steps)
        if abs(force_error) < 2.0:
            self.grasp_stable_counter += 1
        else:
            self.grasp_stable_counter = 0
            
        if self.grasp_stable_counter > 10:
            print("Grasp stabilized, beginning lift")
            self.state = PandaState.LIFT

    def _lift_state(self):
        """Lift object while maintaining grasp"""
        current_pos = p.getLinkState(
            self.controller.panda,
            self.controller.joint_info['panda_hand_joint']['index']
        )[0]
        
        # Target position (lift straight up)
        target_pos = [current_pos[0], current_pos[1], self.lift_height]
        
        # Calculate IK for lift
        lift_joints = p.calculateInverseKinematics(
            self.controller.panda,
            self.controller.joint_info['panda_hand_joint']['index'],
            target_pos,
            targetOrientation=p.getQuaternionFromEuler([0, -np.pi, 0])
        )
        
        # Move arm
        for i in range(7):
            p.setJointMotorControl2(
                self.controller.panda,
                i,
                p.POSITION_CONTROL,
                targetPosition=lift_joints[i],
                force=100
            )
        
        # Maintain grasp force
        current_force = np.linalg.norm(self.contact_force)
        if current_force < 5.0:  # Object dropped
            print("Object dropped during lift!")
            self.state = PandaState.DONE
        elif current_pos[2] > self.lift_height - 0.01:
            print("Lift successful!")
            self.state = PandaState.DONE

    def get_gripper_center(self):
        """Helper to compute midpoint between fingers"""
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