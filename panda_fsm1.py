from enum import Enum, auto
import numpy as np
import pybullet as p
import math

class PandaState(Enum):
    CLOSE = auto()      # Close until contact detected
    GRASP = auto()      # Apply stabilizing force
    LIFT = auto()       # Raise the object
    HOLD = auto()       # Verify stable grasp
    DONE = auto()       # Final state

class PandaFSM:
    def __init__(self, panda_controller):
        self.controller = panda_controller
        self.state = PandaState.CLOSE  # Start immediately with closing
        self.timer = 0
        
        # Control targets
        self.gripper_target = 0.0     # Start closing immediately (0.0 = closed)
        self.gripper_force = 0        
        self.arm_target_pos = None    
        self.arm_target_orn = None    
        
        # Parameters
        self.lift_height = 0.3        # meters
        self.grasp_force = 15.0       # Newtons
        self.contact_threshold = 5.0  # Newtons
        
        # State tracking
        self.initial_height = self.controller.get_gripper_center()[2]
        self.contact_force = np.zeros(3)
        self.stable_counter = 0

    def update(self):
        """Main update loop"""
        self.timer += 1
        current_pos = self.controller.get_gripper_center()
        self.contact_force = self.controller._compute_sph_reaction_force(current_pos)
        force_magnitude = np.linalg.norm(self.contact_force)
        
        # State machine logic
        if self.state == PandaState.CLOSE:
            self._close_state(force_magnitude)
        elif self.state == PandaState.GRASP:
            self._grasp_state(force_magnitude)
        elif self.state == PandaState.LIFT:
            self._lift_state(current_pos)
        elif self.state == PandaState.HOLD:
            self._hold_state()
        elif self.state == PandaState.DONE:
            return False
        
        self._apply_control()
        return True

    def _close_state(self, force_magnitude):
        """Close gripper until contact"""
        # Already set gripper_target to 0.0 in init
        if force_magnitude > self.contact_threshold:
            print(f"Contact detected ({force_magnitude:.2f}N)")
            self.state = PandaState.GRASP
            # Switch to force control
            self.gripper_force = 0.1  # Initial force

    def _grasp_state(self, force_magnitude):
        """Apply and maintain grasp force"""
        force_error = self.grasp_force - force_magnitude
        self.gripper_force += 0.05 * force_error  # PI-like controller
        self.gripper_force = np.clip(self.gripper_force, 0.1, 2.0)  # Safety limits
        
        if abs(force_error) < 2.0:
            self.stable_counter += 1
        else:
            self.stable_counter = 0
            
        if self.stable_counter > 10:
            print(f"Stable grasp at {force_magnitude:.2f}N")
            self.state = PandaState.LIFT

    def _lift_state(self, current_pos):
        """Lift object to target height"""
        target_height = self.initial_height + self.lift_height
        self.arm_target_pos = [current_pos[0], current_pos[1], current_pos[2] + 0.001]
        self.arm_target_orn = p.getQuaternionFromEuler([0, -math.pi, 0])
        
        # Maintain force during lift
        force_error = self.grasp_force - np.linalg.norm(self.contact_force)
        self.gripper_force += 0.02 * force_error
        
        if current_pos[2] >= target_height - 0.01:
            self.state = PandaState.HOLD

    def _hold_state(self):
        """Verify stable hold"""
        if self.timer > 100:  # ~0.5 second hold
            self.state = PandaState.DONE
            print("Operation completed")

    def _apply_control(self):
        """Execute all control commands"""
        # Arm movement
        if self.arm_target_pos:
            joint_positions = p.calculateInverseKinematics(
                self.controller.panda,
                self.controller.end_effector_index,
                self.arm_target_pos,
                self.arm_target_orn,
                maxNumIterations=100
            )
            for i in range(7):
                p.setJointMotorControl2(
                    self.controller.panda, i,
                    p.POSITION_CONTROL,
                    targetPosition=joint_positions[i],
                    force=300,
                    positionGain=0.3
                )
        
        # Gripper control
        for joint_name in ['panda_finger_joint1', 'panda_finger_joint2']:
            if joint_name in self.controller.joint_info:
                joint_idx = self.controller.joint_info[joint_name]['index']
                if self.state in [PandaState.GRASP, PandaState.LIFT, PandaState.HOLD]:
                    p.setJointMotorControl2(
                        self.controller.panda, joint_idx,
                        p.TORQUE_CONTROL,
                        force=-abs(self.gripper_force)  # Always closing force
                    )
                else:  # CLOSE state
                    p.setJointMotorControl2(
                        self.controller.panda, joint_idx,
                        p.POSITION_CONTROL,
                        targetPosition=self.gripper_target,
                        force=10,
                        positionGain=0.1
                    )