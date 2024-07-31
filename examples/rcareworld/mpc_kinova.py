import time

# Third Party
import numpy as np
import torch
from tqdm import tqdm

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig

# rcareworld env
from env import RCareMPC, RCareMPCKinova
from utils import unity2storm


def main():
    tensor_args = TensorDeviceType()
    world_file = "collision_test.yml"

    # specify the robot type for kinova
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), "kinova_gen3.yml"))["robot_cfg"]
    robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)

    # specify mpc params
    mpc_config = MpcSolverConfig.load_from_robot_config(
        robot_cfg,
        world_file,
        use_cuda_graph=True,
        use_cuda_graph_metrics=True,
        use_cuda_graph_full_step=False,
        use_lbfgs=False,
        use_es=False,
        use_mppi=True,
        store_rollouts=True,
        step_dt=0.02,
    )
    mpc = MpcSolver(mpc_config)

    # initialize the environment in RCqareWorld
    env = RCareMPCKinova()

    env.initialize()

    tgt_p = env.get_target_eef_pose()
    unity_cube_position = tgt_p['position']
    env.robot.BioIKMove(unity_cube_position, 0, False)
    env.robot.BioIKRotateQua(taregetEuler=[90, 0, 0], duration=0, relative=False)
    for i in range(10):
        env.step()
    cube_position = unity2storm(position=tgt_p['position'])
    cube_orientation    = unity2storm(orientation=tgt_p['orientation']) 
    print("cube_position: ", cube_position)
    print("cube_orientation: ", cube_orientation)
    past_position = cube_position

    retract_cfg = mpc.rollout_fn.dynamics_model.retract_config.clone().unsqueeze(0)
    joint_names = mpc.rollout_fn.joint_names

    state = mpc.rollout_fn.compute_kinematics(
        JointState.from_position(retract_cfg, joint_names=joint_names)
    )
    current_state = JointState.from_position(retract_cfg, joint_names=joint_names)
    retract_pose = Pose(state.ee_pos_seq, quaternion=state.ee_quat_seq)
    goal = Goal(
        current_state=current_state,
        goal_state=JointState.from_position(retract_cfg, joint_names=joint_names),
        goal_pose=retract_pose,
    )

    goal_buffer = mpc.setup_solve_single(goal, 1)
    mpc.update_goal(goal_buffer)
    mpc_result = mpc.step(current_state, max_attempts=2)
    # print(mpc_result)

    cmd_state_full = None

    for i in tqdm(range(50)):
        # get the position of the cube
        tgt_p = env.get_target_eef_pose()
        cube_position = unity2storm(position=tgt_p['position'])
        cube_orientation    = unity2storm(orientation=tgt_p['orientation'])

        if np.linalg.norm(cube_position-past_position) > 0.001:
            ee_translation_goal = cube_position
            ee_orientation_teleop_goal = cube_orientation
            ik_goal = Pose(
                position=tensor_args.to_device(ee_translation_goal),
                quaternion=tensor_args.to_device(ee_orientation_teleop_goal),
            )
            goal_buffer.goal_pose.copy_(ik_goal)
            mpc.update_goal(goal_buffer)
            past_pose = cube_position
        
        sim_js = env.get_robot_joint_positions()

        print("sim_js: ", sim_js)

        cu_js = JointState(
            position=tensor_args.to_device(sim_js),
            velocity=tensor_args.to_device(sim_js) * 0.0,
            acceleration=tensor_args.to_device(sim_js) * 0.0,
            jerk=tensor_args.to_device(sim_js) * 0.0,
            joint_names=joint_names,
        )

        if cmd_state_full is None:
            current_state.copy_(cu_js)
        else:
            current_state_partial = cmd_state_full.get_ordered_joint_state(
                mpc.rollout_fn.joint_names
            )
            current_state.copy_(current_state_partial)
            current_state.joint_names = current_state_partial.joint_names
        common_js_names = []
        current_state.copy_(cu_js)

        mpc_result = mpc.step(current_state, max_attempts=2)

        succ = True  # ik_result.success.item()
        cmd_state_full = mpc_result.js_action
        common_js_names = []
        idx_list = []
        for x in sim_js:
            if x in cmd_state_full.joint_names:
                idx_list.append(robot.get_dof_index(x))
                common_js_names.append(x)
        
        # env.robot.BioIKRotateQua(taregetEuler=[90, 0, 0], duration=3, relative=False)
        env.step()
    env.close()
    exit()




if __name__ == "__main__":
    main()
