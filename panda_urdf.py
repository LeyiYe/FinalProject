import pybullet as p
import pybullet_data
import numpy as np

class SimplifiedPandaFSM:
    def __init__(self):
        # Physics client setup
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load plane and Panda hand
        self.plane_id = p.loadURDF("plane.urdf")
        self.panda = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
        
        # Get joint information
        self.num_joints = p.getNumJoints(self.panda)
        self.joint_info = self._get_joint_info()
        
        # FSM state
        self.state = "open"
        self.timer = 0
        
    def _get_joint_info(self):
        joint_info = {}
        for i in range(self.num_joints):
            info = p.getJointInfo(self.panda, i)
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            joint_limit_lower = info[8]
            joint_limit_upper = info[9]
            
            joint_info[joint_name] = {
                'index': i,
                'type': joint_type,
                'limit_lower': joint_limit_lower,
                'limit_upper': joint_limit_upper
            }
        return joint_info
    
    def run(self):
        while True:
            # State machine logic
            if self.state == "open":
                self._open_state()
            elif self.state == "close":
                self._close_state()
            # Add other states as needed
            
            p.stepSimulation()
            self.timer += 1
    
    def _open_state(self):
        # Move fingers to open position
        p.setJointMotorControlArray(
            self.panda,
            [self.joint_info['panda_finger_joint1']['index'], 
            self.joint_info['panda_finger_joint2']['index']],
            p.POSITION_CONTROL,
            targetPositions=[0.04, 0.04]  # Open position
        )
        
        if self.timer > 100:  # After 100 steps
            self.state = "close"
            self.timer = 0
    
    def _close_state(self):
        # Close fingers
        p.setJointMotorControlArray(
            self.panda,
            [self.joint_info['panda_finger_joint1']['index'], 
            self.joint_info['panda_finger_joint2']['index']],
            p.POSITION_CONTROL,
            targetPositions=[0.0, 0.0]  # Closed position
        )
        
        # Check contact forces (simplified)
        contacts = p.getContactPoints(self.panda)
        if len(contacts) > 0:
            self.state = "squeeze"
            self.timer = 0

if __name__ == "__main__":
    fsm = SimplifiedPandaFSM()
    fsm.run()