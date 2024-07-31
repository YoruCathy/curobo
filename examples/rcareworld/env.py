from pyrcareworld.envs import RCareWorld
import pybullet as p
import numpy as np
import math

class RCareMPC(RCareWorld):
    
    def __init__(
        self,
        executable_file: str = None,
        scene_file: str = None,
        custom_channels: list = [],
        assets: list = [],
        **kwargs
    ):
        RCareWorld.__init__(
            self,
            executable_file=executable_file,
            scene_file=scene_file,
            custom_channels=custom_channels,
            assets=assets,
            **kwargs,
        )

        self.robot_id = 639787
        self.robot_dof = 7

        # uncomment below lines to recreate isaac sim environment

        # sphere_1 = self.create_object(id=10001, name="Sphere", is_in_scene=True)
        # sphere_1.setTransform(position=[0.4, 0.1, 0.4],scale=[0.2, 0.2, 0.2])

        # cube_1 = self.create_object(id=20001, name="Cube", is_in_scene=False)
        # cube_1.load()
        # cube_1.setTransform(position=[0.4, 0.2, 0.2],scale=[0.3, 0.4, 0.1])

        # cube_2 = self.create_object(id=20002, name="Cube", is_in_scene=False)
        # cube_2.load()
        # cube_2.setTransform(position=[ 0.4,  0.2, -0.3],scale=[0.3, 0.5, 0.1])

        cube_3 = self.create_object(id=20003, name="Cube", is_in_scene=False)
        cube_3.load()
        cube_3.setTransform(position=[ 0. , -0.1,  0. ],scale=[2.0, 0.2, 2.0])

        #define pose of reference cube
        cube_ref_pose = [-0.3, 0.5, 0.0]

        cube_ref = self.create_object(id=30001, name="Cube", is_in_scene=True)
        # cube_ref.load()
        cube_ref.setTransform(position=cube_ref_pose,scale=[0.01, 0.01, 0.01])

        robot_id = 639787
        robot_dof = 7
        robot = self.create_robot(
            id=robot_id, 
            gripper_list=["6397870"], 
            robot_name="franka_panda",
            base_pos=[0, 0, 0]
        )
        self._step()
        print("Initialized RCareStorm object!")

        # print(robot.getRobotState())

        # self.stepSeveralSteps(50)

        self.instance_channel.set_action(
            "IKTargetDoMove",
            id=639787,
            position=cube_ref_pose,
            duration=0,
            speed_based=False,
        )
        print("Moved robot to reference cube!")
        self.instance_channel.set_action(
            "IKTargetDoRotate",
            id=639787,
            vector3=[0, 0, 0],
            duration=0,
            speed_based=False,
        )
        print("Rotated robot to reference cube!")

    def get_robot_joint_positions(self):
        return self.instance_channel.data[self.robot_id]['joint_positions']

    def get_robot_joint_velocities(self):
        return self.instance_channel.data[self.robot_id]['joint_velocities']

    def get_robot_joint_accelerations(self):
        return self.instance_channel.data[self.robot_id]['joint_accelerations']

    def get_target_eef_pose(self):
        target_pose = {}
        target_pose['position'] = self.instance_channel.data[30001]['position']
        target_pose['orientation'] = self.instance_channel.data[30001]['rotation']
        return target_pose
        
    def set_robot_joint_position(self, joint_positions=None, joint_velocities=None, joint_accelerations=None):
        
        if joint_positions is not None:
            self.instance_channel.set_action(
                'SetJointPositionDirectly',
                id=self.robot_id,
                joint_positions=list(joint_positions[0:self.robot_dof]),
            )
        
        if joint_velocities is not None:
            self.instance_channel.set_action(
                'SetJointVelocity',
                id=self.robot_id,
                joint_velocitys=list(joint_velocities[0:self.robot_dof]),
            )

        if joint_accelerations is not None:
            self.instance_channel.set_action(
                    'SetJointAcceleration',
                    id=self.robot_id,
                    joint_accelerations=list(joint_accelerations[0:self.robot_dof]),
                )

    def set_eef_pose(self, pos: list, rot: list):
        q = self.robot.ik_controller.calculate_ik_recursive(pos, eef_orn=p.getQuaternionFromEuler(rot))
        self.set_robot_joints(joint_positions=joint_positions)

    def step(self):
        self._step()

class RCareMPCKinova(RCareWorld):
    
    def __init__(
        self,
        executable_file: str = None,
        scene_file: str = None,
        custom_channels: list = [],
        assets: list = [],
        **kwargs
    ):
        RCareWorld.__init__(
            self,
            executable_file=executable_file,
            scene_file=scene_file,
            custom_channels=custom_channels,
            assets=assets,
            **kwargs,
        )

        self.robot_id = 315893
        self.robot_dof = 7
        self.robot = self.create_robot(
            id=self.robot_id, 
            gripper_list=[str(self.robot_id*10)], 
            robot_name="kinova_gen3_7dof",
            base_pos=[0, 0, 0]
        )
        print("Initialized RCareStorm Kinova object!")
    
    def initialize(self):
        self.cube_ref = self.create_object(id=30001, name="Cube", is_in_scene=True)
        cube_ref_pose = self.cube_ref.getPosition()
        self._step()


        self.instance_channel.set_action(
            "IKTargetDoMove",
            id=self.robot_id,
            position=cube_ref_pose,
            duration=0,
            speed_based=False,
        )

        # self.robot.BioIKRotateQua(taregetEuler=[90, 0, 0], duration=30, relative=False)
        self._step()
        print("Moved robot to reference cube!")

    def get_robot_joint_positions(self):
        return self.instance_channel.data[self.robot_id]['joint_positions']

    def get_robot_joint_velocities(self):
        return self.instance_channel.data[self.robot_id]['joint_velocities']

    def get_robot_joint_accelerations(self):
        return self.instance_channel.data[self.robot_id]['joint_accelerations']

    def get_target_eef_pose(self):
        target_pose = {}
        target_pose['position'] = self.instance_channel.data[30001]['position']
        target_pose['orientation'] = self.instance_channel.data[30001]['rotation']
        return target_pose
        
    def set_robot_joint_position(self, joint_positions=None, joint_velocities=None, joint_accelerations=None):
        
        if joint_positions is not None:
            self.instance_channel.set_action(
                'SetJointPositionDirectly',
                id=self.robot_id,
                joint_positions=list(joint_positions[0:self.robot_dof]),
            )
        
        if joint_velocities is not None:
            self.instance_channel.set_action(
                'SetJointVelocity',
                id=self.robot_id,
                joint_velocitys=list(joint_velocities[0:self.robot_dof]),
            )

        if joint_accelerations is not None:
            self.instance_channel.set_action(
                    'SetJointAcceleration',
                    id=self.robot_id,
                    joint_accelerations=list(joint_accelerations[0:self.robot_dof]),
                )

    def set_eef_pose(self, pos: list, rot: list):
        q = self.robot.ik_controller.calculate_ik_recursive(pos, eef_orn=p.getQuaternionFromEuler(rot))
        self.set_robot_joints(joint_positions=joint_positions)

    def step(self):
        self._step()

if __name__=="__main__":
    env = RCareMPC(executable_file="put path to executable here")
    env.initialize()
    env.step()
    env.close()
    exit()
    # env = RCareMPCKinova()
    # env.initialize()
    # env.step()
    # env.close