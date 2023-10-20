import numpy as np
from widowx_envs.utils.grasp_utils import rgb_to_robot_coords
from widowx_envs.utils.params import *


class PickPlacePolicy:

    def __init__(self, env, pick_height_mean=0.12, pick_height_std=0.03,
                lift_height_after_drop_mean=0.06, lift_height_after_drop_std=0.02,
                xyz_action_scale=0.3, pick_z_multiplier=1.0,
                far_reach_z_thresh=np.inf, far_reach_z_penalty=0.2, wrist_action_scale=1.0,
                pitch_action_scale=0.5, roll_action_scale=0.5,
                wrist_angle_range=(-1., 1.), wrist_target_thresh=0.05, height_thresh=0.015,
                xy_thresh = 0.02, gripper_thresh = 0.5, 
                gripper_open_time_horizon=10, height_open_gripper=0.08,
                pick_point_offset=(0, 0, 0),
                **kwargs):
        self.env = env
        self.pick_height_mean = pick_height_mean
        self.pick_height_std = pick_height_std
        self.lift_height_after_drop_mean = lift_height_after_drop_mean
        self.lift_height_after_drop_std = lift_height_after_drop_std
        self.xyz_action_scale = xyz_action_scale
        self.pick_z_multiplier = pick_z_multiplier

        #Avoid IK errors when reaching high/far away from neutral
        self.far_reach_z_thresh = far_reach_z_thresh
        self.far_reach_z_penalty = far_reach_z_penalty

        if self.env._hp.action_mode == '3trans3rot':
            self.wrist_angle_index = 5
        else:
            self.wrist_angle_index = 4
        self.wrist_action_scale = wrist_action_scale
        self.pitch_action_scale = pitch_action_scale
        self.roll_action_scale = roll_action_scale
        self.wrist_angle_range = wrist_angle_range
        self.wrist_target_thresh = wrist_target_thresh
        self.height_thresh = height_thresh
        self.xy_thresh = xy_thresh
        self.gripper_thresh = gripper_thresh
        self.gripper_open_time_horizon = gripper_open_time_horizon
        self.height_open_gripper = height_open_gripper
        self.current_gripper = 1.0
        self.previous_gripper = 1.0
        self.pick_point_offset = pick_point_offset

    def reset(self, pick_point, drop_point, initial_gripper):
        self.pick_point = pick_point + self.pick_point_offset
        self.drop_point = drop_point
        self.grasp_executed = False
        self.wrist_target_achieved = False
        self.ungrasp_executed = False
        self.move_to_random_end_pos = False
        self.wrist_target_pick = np.random.rand() * (self.wrist_angle_range[1] - self.wrist_angle_range[0]) + self.wrist_angle_range[0]
        self.pick_height = min(np.random.normal(loc=self.pick_height_mean, scale=self.pick_height_std), WORKSPACE_BOUNDARIES[1][2] - 0.02)
        self.lift_height_after_drop = np.random.normal(loc=self.lift_height_after_drop_mean, scale=self.lift_height_after_drop_std)
        self.current_gripper = initial_gripper
        self.previous_gripper = initial_gripper
        self.time_to_open_gripper = np.random.choice(np.arange(self.gripper_open_time_horizon))
        self.end_random_pos = np.random.uniform(self.env._low_bound[:3] + np.array([0, 0, 0.07]), self.env._high_bound[:3] - np.array([0, 0.07, 0]))
        self.status = ""

    def get_action(self, obs=None, curr_ts=0):
        if obs is None:
            obs = self.env._get_obs()
        ee_pos = obs['state'][:3]
        wrist_angle = obs['state'][self.wrist_angle_index]
        done = False
        gripper_droppoint_xy_dist = np.linalg.norm(self.drop_point[:2] - ee_pos[:2])
        gripper_random_endpos_xy_dist = np.linalg.norm(self.end_random_pos[:2] - ee_pos[:2])
        wrist_dist = abs(self.wrist_target_pick - wrist_angle)
        gripper_state = obs['state'][-1]
        self.previous_gripper = self.current_gripper
        self.current_gripper = gripper_state
        gripper_moving = not (self.previous_gripper - 0.01 < self.current_gripper < self.previous_gripper + 0.01)

        gripper_pickpoint_xy_dist = np.linalg.norm(self.pick_point[:2] - ee_pos[:2])

        if (gripper_pickpoint_xy_dist > self.xy_thresh or (self.env._hp.action_mode == '3trans1rot' and
            wrist_dist > self.wrist_target_thresh and not self.wrist_target_achieved)) and not self.grasp_executed \
            and self.status != 'approaching_close':
            print('moving near obj')
            action_xyz = (self.pick_point - ee_pos) * self.xyz_action_scale
            action_xyz[2] = 0.0
            if ee_pos[2] > self.far_reach_z_thresh:
                action_xyz[2] -= self.far_reach_z_penalty
            action_xyz[2] *= self.pick_z_multiplier
            if curr_ts < self.time_to_open_gripper and ee_pos[2] > self.height_open_gripper:
                action_gripper = [self.current_gripper]
            else:
                action_gripper = [1.0]
            action_wrist = [(self.wrist_target_pick - wrist_angle) * self.wrist_action_scale]
            self.status = 'approaching'
        elif ee_pos[2] - self.pick_point[2] > self.height_thresh and not self.grasp_executed:
            print('moving near obj, down')
            action_xyz = (self.pick_point - ee_pos) * self.xyz_action_scale
            if curr_ts < self.time_to_open_gripper and ee_pos[2] > self.height_open_gripper:
                action_gripper = [self.current_gripper]
            else:
                action_gripper = [1.0]
            action_wrist = [(self.wrist_target_pick - wrist_angle) * self.wrist_action_scale]
            self.wrist_target_achieved = True
            self.status = 'approaching_close'
        elif not self.grasp_executed or (gripper_moving and not self.ungrasp_executed):
            print('executing grasp')
            action_xyz = [0., 0., 0.]
            action_gripper = [0.]
            action_wrist = [(self.wrist_target_pick - wrist_angle) * self.wrist_action_scale]
            self.status = 'grasping'
            self.grasp_executed = True
        elif self.pick_height - ee_pos[2] > self.height_thresh and self.status != 'approaching_drop' and not self.ungrasp_executed:
            print('lifting')
            action_xyz = (STARTPOS - ee_pos) * self.xyz_action_scale
            action_xyz[0] = 0.0
            action_xyz[1] = 0.0
            action_xyz[2] -= 0.02 * self.xyz_action_scale
            action_gripper = [0.]
            action_wrist = [(0.0 - wrist_angle) * self.wrist_action_scale]
            self.status = 'lifting'
        elif (gripper_droppoint_xy_dist > self.xy_thresh
                and self.grasp_executed and not self.ungrasp_executed):
            print("approaching drop")
            action_xyz = (self.drop_point - ee_pos) * self.xyz_action_scale
            action_xyz[2] = 0.0
            action_gripper = [0.]
            action_wrist = [(0.0 - wrist_angle) * self.wrist_action_scale]
            self.status = 'approaching_drop'
        elif ((ee_pos[2] - self.drop_point[2] > self.height_thresh)
                and self.grasp_executed and not self.ungrasp_executed):
            print("lowering for drop")
            action_xyz = (self.drop_point - ee_pos) * self.xyz_action_scale
            action_gripper = [0.]
            action_wrist = [(0.0 - wrist_angle) * self.wrist_action_scale]
            self.status = 'approaching_drop'
        elif not self.ungrasp_executed and not gripper_moving:
            print("ungrasping")
            action_xyz = [0., 0., 0.]
            action_gripper = [1.0]
            self.ungrasp_executed = True
            action_wrist = [(0.0 - wrist_angle) * self.wrist_action_scale]
            self.status = 'un-grasping'
        elif self.lift_height_after_drop - ee_pos[2] > self.height_thresh:
            print("lifting after drop")
            self.ungrasp_executed = True
            action_xyz = (STARTPOS - ee_pos) * self.xyz_action_scale
            action_xyz[0] = 0.0
            action_xyz[1] = 0.0
            action_xyz[2] -= 0.02 * self.xyz_action_scale
            action_gripper = [1.0]
            action_wrist = [(0.0 - wrist_angle) * self.wrist_action_scale]
            self.status = 'lifting_after_drop'
        elif gripper_random_endpos_xy_dist > self.xy_thresh and not self.move_to_random_end_pos:
            print('gripper move to random end position')
            action_xyz = (self.end_random_pos - ee_pos) * self.xyz_action_scale
            action_gripper = [1.0]
            action_wrist = [(0.0 - wrist_angle) * self.wrist_action_scale]
        else:
            print("done")
            self.move_to_random_end_pos = True
            action_xyz = [0., 0., 0.]
            action_gripper = [1.0]
            action_wrist = [(0.0 - wrist_angle) * self.wrist_action_scale]
            self.status = 'done'

        agent_info = dict(done=done, status=self.status)
        if self.env._hp.action_mode == '3trans':
            action = np.concatenate((action_xyz, action_gripper))
        elif self.env._hp.action_mode == '3trans1rot':
            action = np.concatenate((action_xyz, action_wrist, action_gripper))
        elif self.env._hp.action_mode == '3trans3rot':
            # corrective actions for pitch and roll because of noise
            # pitch_angle = obs['state'][3]
            # roll_angle = obs['state'][4]
            # action_pitch = [(0. - pitch_angle) * self.pitch_action_scale]
            # action_roll = [(0. - roll_angle) * self.roll_action_scale]
            
            action_pitch = [0]
            action_roll = [0]

            action = np.concatenate((action_xyz, action_pitch, action_roll, action_wrist, action_gripper))

        return action, agent_info