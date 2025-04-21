from enum import Enum, auto
import numpy as np

class PandaState(Enum):
    OPEN = auto()
    APPROACH = auto()     
    CLOSE = auto()
    START_CLOSER = auto()
    CLOSE_SOFT = auto()
    GRASP = auto()         
    LIFT = auto()          
    SQUEEZE = auto()
    SQUEEZE_HOLDING = auto()
    SQUEEZE_NO_GRAVITY = auto()
    HANG = auto()
    REORIENT = auto()
    LIN_ACC = auto()
    ANG_ACC = auto()
    DONE = auto()

class PandaFSM:
    def __init__(self, panda_controller):
        self.state = PandaState.OPEN
        self.controller = panda_controller
        self.timer = 0
        self._init_variables()

    def _init_variables(self):
        """Initialize all state variables"""
        # Contact and force tracking
        self.contacts = []
        self.F_history = []
        self.filtered_forces = []
        self.f_moving_average = []
        self.f_errs = np.ones(10)
        
        # Gripper state
        self.left_has_contacted = False
        self.right_has_contacted = False
        self.franka_positions_at_contact = np.zeros(self.controller.config['franka']['num_joints'])
        self.grippers_pre_squeeze = [-1, -1]
        
        # Counters
        self.close_fails = 0
        self.squeeze_counter = 0
        self.squeeze_holding_counter = 0
        self.squeeze_no_gravity_counter = 0
        self.open_counter = 0
        self.hang_counter = 0
        
        # Control outputs
        self.vel_des = np.zeros(self.controller.config['franka']['num_joints'])
        self.pos_des = np.zeros(self.controller.config['franka']['num_joints'])
        self.torque_des = np.zeros(self.controller.config['franka']['num_joints'])
        self.running_torque = [-0.1, -0.1]
        
        # Success flags
        self.pickup_success = False

    def update(self):
        """Update the state machine"""
        self.timer += 1
        
        F_curr = self.controller.get_gripper_force()
        gripper_pos = self.controller.get_gripper_positions()
        
        if self.state == PandaState.OPEN:
            self._open_state()
        elif self.state == PandaState.APPROACH:
            self._approach_state()
        elif self.state == PandaState.GRASP:
            self._grasp_state(F_curr)
        elif self.state == PandaState.LIFT:
            self._lift_state()
        elif self.state == PandaState.CLOSE:
            self._close_state(F_curr, gripper_pos)
        elif self.state == PandaState.START_CLOSER:
            self._start_closer_state(gripper_pos)
        elif self.state == PandaState.CLOSE_SOFT:
            self._close_soft_state(F_curr)
        elif self.state == PandaState.SQUEEZE:
            self._squeeze_state(F_curr, gripper_pos)
        elif self.state == PandaState.SQUEEZE_HOLDING:
            self._squeeze_holding_state()
        elif self.state == PandaState.SQUEEZE_NO_GRAVITY:
            self._squeeze_no_gravity_state(F_curr, gripper_pos)
        elif self.state == PandaState.HANG:
            self._hang_state(F_curr)
        elif self.state == PandaState.DONE:
            self.controller.set_gripper_velocity(0, 0)
            self.controller.set_gripper_torque(0, 0)
            return False  # Signal to stop
        
        return True  # Continue running

    def _open_state(self):
        """Open gripper state"""
        print("OPEN STATE - Moving gripper open")
        self.open_counter += 1
        
        self.controller.set_gripper_velocity(0.5, 0.5)
        
        if self.open_counter > 100:
            self.state = PandaState.CLOSE
            self.open_counter = 0

    def _approach_state(self):
        """Move gripper above the object"""
        target_pos = [0, 0, 0.15]  # 15cm above object center
        current_pos = self.controller.get_gripper_center()
        
        # Calculate target joint positions using inverse kinematics
        target_joint_positions = p.calculateInverseKinematics(
            self.controller.panda,
            self.controller.joint_info['panda_hand_joint']['index'],
            target_pos,
            targetOrientation=p.getQuaternionFromEuler([0, -np.pi, 0])
        )
        
        # Move arm using position control
        for i in range(7):  # First 7 joints are the arm
            p.setJointMotorControl2(
                self.controller.panda,
                i,
                p.POSITION_CONTROL,
                targetPosition=target_joint_positions[i],
                force=100
            )
        
        # Open gripper during approach
        self.controller.set_gripper_velocity(0.5, 0.5)
        
        if np.linalg.norm(np.array(target_pos) - np.array(current_pos)) < 0.02:
            print("Approach complete, switching to GRASP state")
            self.state = PandaState.GRASP

    def _grasp_state(self, F_curr):
        """Close gripper to grasp object with force control"""
        target_force = self.config['force_control']['grasp_force']
        
        # Simple P controller for grasp force
        force_error = target_force - sum(F_curr)
        Kp = self.config['force_control']['Kp']
        
        # Adjust torque based on force error
        self.running_torque[0] -= min(force_error * Kp, 3 * Kp)
        self.running_torque[1] -= min(force_error * Kp, 3 * Kp)
        self.set_gripper_torque(self.running_torque[0], self.running_torque[1])
        
        # Check if we've achieved sufficient grasp force
        if sum(F_curr) > target_force * 0.9:  # 90% of target force
            print(f"Grasped with force {sum(F_curr):.2f}N, switching to LIFT")
            self.state = PandaState.LIFT

    def _lift_state(self):
        """Lift the grasped object"""
        target_height = self.config['franka']['lift_height']
        current_pos = p.getLinkState(self.panda, self.joint_info['panda_hand_joint']['index'])[0]
        
        # Calculate IK for lifted position
        lifted_pos = [current_pos[0], current_pos[1], target_height]
        target_joint_positions = p.calculateInverseKinematics(
            self.panda,
            self.joint_info['panda_hand_joint']['index'],
            lifted_pos,
            targetOrientation=p.getQuaternionFromEuler([0, -np.pi, 0])
        )
        
        # Move arm up while maintaining grasp
        for i in range(7):  # Arm joints
            p.setJointMotorControl2(
                self.panda,
                i,
                p.POSITION_CONTROL,
                targetPosition=target_joint_positions[i],
                force=100
            )
        
        # Maintain grasp force
        self.set_gripper_torque(self.running_torque[0], self.running_torque[1])
        
        # Transition when reached height
        if current_pos[2] > target_height - 0.01:
            print("Lift complete, switching to HANG state")
            self.state = PandaState.HANG



    def _close_state(self, F_curr, gripper_pos):
        """Close gripper until contact"""
        print("CLOSE STATE - Moving gripper close")
        closing_speed = -0.7
        
        # Close fingers until contact
        if F_curr[0] > 0:
            self.left_has_contacted = True
        if F_curr[1] > 0:
            self.right_has_contacted = True
            
        left_vel = closing_speed if not self.left_has_contacted else 0
        right_vel = closing_speed if not self.right_has_contacted else 0
        
        self.controller.set_gripper_velocity(left_vel, right_vel)
        
        # Check if closed too far without contact
        if (not (self.left_has_contacted or self.right_has_contacted) and 
            all(p < 0.001 for p in gripper_pos)):
            print("Failed: Grippers closed without contacting object.")
            self.state = PandaState.DONE
            
        # Transition when both fingers have contacted
        if self.left_has_contacted and self.right_has_contacted:
            self.franka_positions_at_contact = gripper_pos
            self.state = PandaState.START_CLOSER
    
    def _start_closer_state(self, gripper_pos):
        """Reset and close to near-contact position"""
        print("STAR CLOSING STATE")
        
        # Close until near contact position
        left_vel = closing_speed if gripper_pos[0] < self.franka_positions_at_contact[0] + 0.003 else 0
        right_vel = closing_speed if gripper_pos[1] < self.franka_positions_at_contact[1] + 0.003 else 0
        
        self.controller.set_gripper_velocity(left_vel, right_vel)
        
        # Transition when close to contact position
        if all(p < pos + 0.004 for p, pos in zip(gripper_pos, self.franka_positions_at_contact)):
            self.state = PandaState.CLOSE_SOFT
    
    def _close_soft_state(self, F_curr):
        """Soft closing with force checking"""
        print("CLOSE SOFT STATE")
        # Adjust closing speed based on failures
        first_speed = 0.25
        closing_speed = -first_speed / (self.close_fails + 1)
        
        # Check if forces are too high
        if sum(F_curr) > 300:
            self.close_fails += 1
            self.left_has_contacted = False
            self.right_has_contacted = False
            print("Forces too high during close_soft, resetting state")
            self.state = PandaState.CLOSE
        
        # Check for contact
        force_threshold = 0.005
        left_in_contact = F_curr[0] > force_threshold
        right_in_contact = F_curr[1] > force_threshold
        
        left_vel = closing_speed if not left_in_contact else 0
        right_vel = closing_speed if not right_in_contact else 0
        
        self.controller.set_gripper_velocity(left_vel, right_vel)
        
        # Transition when both fingers have contacted
        if left_in_contact and right_in_contact:
            self.grippers_pre_squeeze = self.get_gripper_positions()
            self.state = PandaState.SQUEEZE
    
    def _squeeze_state(self, F_curr, gripper_pos):
        """Squeeze to desired force"""
        print("SQUEEZE STATE")
        self.squeeze_counter += 1
        
        # Torque control to achieve desired force
        desired_force = 10.0  # Example value
        F_des = np.array([desired_force / 2.0, desired_force / 2.0])
        
        # Simple P controller for torque
        total_F_curr = sum(F_curr)
        total_F_err = sum(F_des) - total_F_curr
        
        Kp = self.config['force_control']['Kp']
        min_torque = self.config['force_control']['min_torque']
        
        self.running_torque[0] -= min(total_F_err * Kp, 3 * Kp)
        self.running_torque[1] -= min(total_F_err * Kp, 3 * Kp)
        self.running_torque[0] = max(min_torque, self.running_torque[0])
        self.running_torque[1] = max(min_torque, self.running_torque[1])
        
        self.set_gripper_torque(self.running_torque[0], self.running_torque[1])
        
        # Check for failure conditions
        if any(p > 0.04 for p in gripper_pos):  # Exceed joint limits
            print("Grippers exceeded joint limits")
            self.state = PandaState.DONE
        
        # Check if desired force is achieved
        if abs(total_F_err) < 0.05 * desired_force:
            print("Desired squeezing force achieved")
            self.state = PandaState.HANG
    
    def _hang_state(self, F_curr):
        """Hang object to test grasp"""
        print("HANG STATE")
        self.hang_counter += 1
        
        # Maintain force with torque control
        desired_force = 10.0  # Example value
        F_des = np.array([desired_force / 2.0, desired_force / 2.0])
        
        total_F_curr = sum(F_curr)
        total_F_err = sum(F_des) - total_F_curr
        
        Kp = self.config['force_control']['Kp']
        self.running_torque[0] -= total_F_err * Kp
        self.running_torque[1] -= total_F_err * Kp
        
        self.set_gripper_torque(self.running_torque[0], self.running_torque[1])
        
        # Check if object is still held
        if total_F_curr < 1.0:  # Object dropped
            self.pickup_success = False
            print("Object dropped during hang")
            self.state = PandaState.DONE
        elif self.hang_counter > 200:  # Successfully held
            self.pickup_success = True
            print("Pickup successful")
            self.state = PandaState.DONE

    def _squeeze_holding_state(self):
        """Holding squeeze with gradually increasing torque"""
        print("SQUEEZE HOLDING STATE")
        self.squeeze_holding_counter += 1
        
        # Gradually increase torque
        if self.squeeze_holding_counter % 10 == 0:
            self.running_torque[0] -= 0.01
            self.running_torque[1] -= 0.01
        
        self.set_gripper_torque(self.running_torque[0], self.running_torque[1])
        
        # Check for contact
        F_curr = self.get_gripper_force()
        if all(f > 0 for f in F_curr):
            self.state = PandaState.SQUEEZE_NO_GRAVITY
    
    def _squeeze_no_gravity_state(self, F_curr, gripper_pos):
        """Squeeze without gravity effects"""
        print("SQUEEZE NO GRAVITY STATE")
        self.squeeze_no_gravity_counter += 1
        
        # Periodically increase torque
        if self.squeeze_no_gravity_counter % self.config['squeeze_no_gravity']['torque_step_period'] == 0:
            torque_step = self.config['squeeze_no_gravity']['soft_object_torque_step']
            self.running_torque[0] -= torque_step
            self.running_torque[1] -= torque_step
        
        self.set_gripper_torque(self.running_torque[0], self.running_torque[1])
        
        # Check termination conditions
        desired_force = self.config['squeeze_no_gravity']['soft_object_F_des']
        if sum(F_curr) > desired_force or self.squeeze_no_gravity_counter > 500:
            print("Squeeze no gravity complete")
            self.state = PandaState.DONE
   