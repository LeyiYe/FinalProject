import time
import numpy as np
import math
from deformable_object import DeformableObjectSim


useNullSpace = 1
ikSolver = 0
pandaEndEffectorIndex = 11 #8
pandaNumDofs = 7

ll = [-7]*pandaNumDofs
#upper limits for null space (todo: set them to proper range)
ul = [7]*pandaNumDofs
#joint ranges for null space (todo: set them to proper range)
jr = [7]*pandaNumDofs
#restposes for null space
jointPositions=[0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]
rp = jointPositions

class PandaSim(object):
    def __init__(self, bullet_client, offset):
        self.bullet_client = bullet_client
        self.bullet_client.setPhysicsEngineParameter(solverResidualThreshold=0)
        self.offset = np.array(offset)
    
        #print("offset=",offset)
        flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        self.bullet_client.loadURDF("tray/traybox.urdf", [0+offset[0], 0+offset[1], -0.6+offset[2]], [-0.5, -0.5, -0.5, 0.5], flags=flags)
        
                
        # Initialize SPH deformable object
        self.sph_system = DeformableObjectSim(particle_radius=0.005)  # Smaller particles
        self.sph_particles = self.sph_system.create_particles()[0]
        self.sph_solver = self.sph_system.create_solver()

        self._position_sph_object()

        self._create_sph_visualization()
        
        orn=[-0.707107, 0.0, 0.0, 0.707107]#p.getQuaternionFromEuler([-math.pi/2,math.pi/2,0])
        eul = self.bullet_client.getEulerFromQuaternion([-0.5, -0.5, -0.5, 0.5])
        self.panda = self.bullet_client.loadURDF("franka_panda/panda.urdf", np.array([0,0,0])+self.offset, orn, useFixedBase=True, flags=flags)
        index = 0
        self.state = 0
        self.control_dt = 1./240.
        self.finger_target = 0
        self.gripper_height = 0.2
        #create a constraint to keep the fingers centered
        c = self.bullet_client.createConstraint(self.panda,
                        9,
                        self.panda,
                        10,
                        jointType=self.bullet_client.JOINT_GEAR,
                        jointAxis=[1, 0, 0],
                        parentFramePosition=[0, 0, 0],
                        childFramePosition=[0, 0, 0])
        self.bullet_client.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)
    
        for j in range(self.bullet_client.getNumJoints(self.panda)):
            self.bullet_client.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)
            info = self.bullet_client.getJointInfo(self.panda, j)
            #print("info=",info)
            jointName = info[1]
            jointType = info[2]
            if (jointType == self.bullet_client.JOINT_PRISMATIC):
                self.bullet_client.resetJointState(self.panda, j, jointPositions[index]) 
                index=index+1

            if (jointType == self.bullet_client.JOINT_REVOLUTE):
                self.bullet_client.resetJointState(self.panda, j, jointPositions[index]) 
                index=index+1
        self.t = 0.

    def _position_sph_object(self):
        """Center SPH object on platform"""
        platform_center = np.array([0, 0.05, -0.6])  # Above platform center
            
            # Calculate particle bounds
        min_x, max_x = np.min(self.sph_particles.x), np.max(self.sph_particles.x)
        min_y, max_y = np.min(self.sph_particles.y), np.max(self.sph_particles.y)
        min_z = np.min(self.sph_particles.z)
            
            # Calculate required offsets
        x_offset = platform_center[0] - (min_x + max_x)/2
        y_offset = platform_center[1] - (min_y + max_y)/2
        z_offset = platform_center[2] - min_z  # Align bottom with platform
            
            # Apply offsets
        self.sph_particles.x += x_offset
        self.sph_particles.y += y_offset
        self.sph_particles.z += z_offset

    def _create_sph_visualization(self):
        """Create visual representation of SPH particles in PyBullet"""
        self.sph_visuals = []
        particle_shape = self.bullet_client.createVisualShape(
        self.bullet_client.GEOM_SPHERE,
        radius=self.sph_system.particle_radius,
        rgbaColor=[1, 0, 0, 0.7]  # Slightly transparent
        )

        particles = self.sph_solver.particles

        for i in range(len(particles[0].x)):
            # Create visual only - no physics
            visual = self.bullet_client.createMultiBody(
                baseMass=0,  # Massless
                baseVisualShapeIndex=particle_shape,
                basePosition=[
                    particles[0].x[i],
                    particles[0].y[i],
                    particles[0].z[i]
                ],
                baseCollisionShapeIndex=-1  # No collision
            )
            self.sph_visuals.append(visual)


    # def _update_sph_visualization(self):
    #     """Update particle positions in visualization"""
    #     particles = self.sph_solver.particles.array[0]
    #     for i, visual in enumerate(self.sph_visuals):
    #         self.bullet_client.resetBasePositionAndOrientation(
    #                     visual,
    #                     posObj=[particles.x[i], particles.y[i], particles.z[i]],
    #                     ornObj=[0, 0, 0, 1]
    #                 )

    def _update_sph_visualization(self):
        """Update particle positions in visualization"""
        # Get current positions from SPH system
        x, y, z = self.sph_particles.x, self.sph_particles.y, self.sph_particles.z
        
        for i, visual in enumerate(self.sph_visuals):
            self.bullet_client.resetBasePositionAndOrientation(
                visual,
                posObj=[x[i], y[i], z[i]],
                ornObj=[0, 0, 0, 1]
            )

    def reset(self):
        pass

    def update_state(self):
        # Update SPH simulation
        for _ in range(5):
            self.sph_system.step()
            
        # Handle coupling between Panda and SPH object
        self._handle_sph_coupling()
            
        # Update visualization
        self._update_sph_visualization()

        # keys = self.bullet_client.getKeyboardEvents()
        # if len(keys)>0:
        #     for k,v in keys.items():
        #         if v&self.bullet_client.KEY_WAS_TRIGGERED:
        #             if (k==ord('0')):
        #                 self.state = 0
        #             if (k==ord('1')):
        #                 self.state = 1
        #             if (k==ord('2')):
        #                 self.state = 2
        #             if (k==ord('3')): # 物体上方 预抓取位置
        #                 self.state = 3
        #             if (k==ord('4')): # 物体抓取位置
        #                 self.state = 4
        #             if (k==ord('5')): # 机械手张开
        #                     self.state = 5
        #             if (k==ord('6')): # 机械手闭合
        #                     self.state = 6
        #             if (k==ord('7')): # 机械手闭合
        #                     self.state = 7
        #             if v&self.bullet_client.KEY_WAS_RELEASED:
        #                 self.state = 0


    def _handle_sph_coupling(self):
        """Handle interaction between gripper and SPH object"""
            # Get gripper position and velocity
        gripper_pos = self.get_gripper_center()
        particles = self.sph_particles
        stiffness = 1e4

        gripper_force = np.zeros(3)
            
        # Vectorized interaction calculation
        dx = particles.x - gripper_pos[0]
        dy = particles.y - gripper_pos[1]
        dz = particles.z - gripper_pos[2]
        dist_sq = dx**2 + dy**2 + dz**2
        mask = dist_sq < 0.02**2  # Interaction radius squared
        
        if np.any(mask):
            # Normalized direction vectors
            dist = np.sqrt(dist_sq[mask])
            nx = dx[mask] / dist
            ny = dy[mask] / dist
            nz = dz[mask] / dist
            
            # Spring force (displacement from equilibrium)
            displacement = 0.02 - dist
            force_magnitude = stiffness * displacement
            
            # Apply forces to particles
            particles.u[mask] += force_magnitude * nx * self.sph_system.dt
            particles.v[mask] += force_magnitude * ny * self.sph_system.dt
            particles.w[mask] += force_magnitude * nz * self.sph_system.dt
            
            # Calculate reaction force on gripper
            gripper_force = -np.array([
                np.sum(force_magnitude * nx),
                np.sum(force_magnitude * ny),
                np.sum(force_magnitude * nz)
            ])  

            # Find nearby particles
        # for i in range(len(self.sph_particles.x)):
        #     dist = np.linalg.norm([
        #     self.sph_particles.x[i] - gripper_pos[0],
        #     self.sph_particles.y[i] - gripper_pos[1],
        #     self.sph_particles.z[i] - gripper_pos[2]
        #     ])
        #     if dist < 0.02:
        #         stiffness = 1e4
        #     displacement = np.array([
        #         gripper_pos[0] - self.sph_particles.x[i],
        #         gripper_pos[1] - self.sph_particles.y[i],
        #         gripper_pos[2] - self.sph_particles.z[i]
        #     ])
        #     self.sph_particles.u[i] += stiffness * displacement[0] * self.sph_solver.dt
        #     self.sph_particles.v[i] += stiffness * displacement[1] * self.sph_solver.dt
        #     self.sph_particles.w[i] += stiffness * displacement[2] * self.sph_solver.dt
            
        #     gripper_force += stiffness * displacement * self.sph_particles.m[i]

        self.bullet_client.applyExternalForce(
            self.panda,
            -1,  # Apply to base
            forceObj=gripper_force,
            posObj=gripper_pos,
            flags=self.bullet_client.WORLD_FRAME
        )


    def step(self, graspWidth):
        # Update SPH simulation
        self.update_state()
        # 设置抓取器张开宽度
        if self.state==6:
            self.finger_target = 0.01
        if self.state==5:
            self.finger_target = 0.04 

        sph_center = [
            np.mean(self.sph_particles.x),
            np.mean(self.sph_particles.y),
            np.mean(self.sph_particles.z)  # Use max Z for top of object
        ]    

        #alpha = 0.9 #0.99
        # if self.state==1 or self.state==2 or self.state==3 or self.state==4 or self.state==7:
        #     #gripper_height = 0.034
        #     self.gripper_height = alpha * self.gripper_height + (1.-alpha)*0.03
        #     # print('self.gripper_height = ', self.gripper_height)

        # if self.state == 2 or self.state == 3 or self.state == 7:
        #         self.gripper_height = alpha * self.gripper_height + (1.-alpha)*0.2
        #     # print('self.gripper_height = ', self.gripper_height)
      
        # t = self.t
        # self.t += self.control_dt
      
        # pos = [self.offset[0]+0.2 * math.sin(1.5 * t), self.offset[1]+self.gripper_height, self.offset[2]+-0.6 + 0.1 * math.cos(1.5 * t)] # 圆形位置
        if self.state == 3:
        # 获取红色积木的位置和方向
            pos = [sph_center[0], sph_center[1]+0.05, sph_center[2]]
            self.prev_pos = pos
            self.gripper_height = pos[1]  # Directly set height


        if self.state == 4:

            pos = [sph_center[0], sph_center[1]-0.02, sph_center[2]]
            self.prev_pos = pos
            self.gripper_height = pos[1]
        if self.state == 7:
            pos = self.prev_pos
            pos[1]+=0.005
            self.prev_pos = pos
            self.gripper_height = pos[1]

        else:
            t = self.t
            self.t += self.control_dt
            pos = [
            self.offset[0] + 0.2 * math.sin(1.5 * t),
            self.gripper_height,
            self.offset[2] - 0.6 + 0.1 * math.cos(1.5 * t)
            ]
        orn = self.bullet_client.getQuaternionFromEuler([math.pi/2.,0.,0.])   # 机械手方向
        # 根据目标位置计算关节位置
        jointPoses = self.bullet_client.calculateInverseKinematics(
            self.panda, 
            pandaEndEffectorIndex, 
            pos, 
            orn, 
            ll, 
            ul, 
            jr, 
            rp, 
            maxNumIterations=20)

        for i in range(pandaNumDofs):
            self.bullet_client.setJointMotorControl2(
                self.panda, i, 
                self.bullet_client.POSITION_CONTROL, 
                jointPoses[i],
                force=5 * 240.)

        #target for fingers
        for i in [9,10]:
            self.bullet_client.setJointMotorControl2(
                self.panda, i, 
                self.bullet_client.POSITION_CONTROL,
                self.finger_target ,
                force= 10)

class PandaSimAuto(PandaSim):
    def __init__(self, bullet_client, offset):
        PandaSim.__init__(self, bullet_client, offset)
        self.state_t = 0
        self.cur_state = 0
        self.states=[3, 3, 4, 6, 7]
        self.state_durations=[2, 3, 2, 3, 5]
        self.reached_target = False
        self.target_pos = None
    
    def update_state(self):
        self.state_t += self.control_dt
        # Check if we've reached the target position
        current_pos = self._get_gripper_position()

        self.target_pos = self._get_target_pos_for_state(self.states[self.cur_state])
        
        dist = np.linalg.norm(np.array(current_pos) - np.array(self.target_pos))
        self.reached_target = dist < 0.01  # 1cm threshold

        if self.state_t > self.state_durations[self.cur_state]:
            self.cur_state += 1
            if self.cur_state >= len(self.states):
                self.cur_state = 0
        self.state_t = 0
        self.state=self.states[self.cur_state]
        self.reached_target = False
        #print("self.state=",self.state)

    def _get_target_pos_for_state(self, state):
        # Same position logic as in step()
        sph_center = [np.mean(self.sph_particles.x),
                     np.mean(self.sph_particles.y),
                     np.mean(self.sph_particles.z)]
        
        if state == 3:
            return [sph_center[0], sph_center[1] + 0.05, sph_center[2]]
        elif state == 4:
            return [sph_center[0], sph_center[1] - 0.02, sph_center[2]]
        elif state == 7:
            return [self.prev_pos[0], self.prev_pos[1] + 0.001, self.prev_pos[2]]
        else:
            return [0, 0.2, -0.6]  # Default home position
        

    def _get_gripper_position(self):
        """Wrapper method to access parent's gripper position"""
        # Get end effector state
        link_state = self.bullet_client.getLinkState(
            self.panda, 
            pandaEndEffectorIndex
        )
        return list(link_state[0])