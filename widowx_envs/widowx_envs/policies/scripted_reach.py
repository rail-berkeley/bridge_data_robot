import numpy as np

class ReachPolicy:
    def __init__(self, env, reach_point,):
        self.env = env
        self.reach_point = reach_point

    def reset(self, reach_point=None):
        if reach_point is not None:
            self.reach_point = reach_point

    def get_action(self):
        ee_pos = self.env._get_obs()['ee_coord']
        action_xyz = self.reach_point - ee_pos
        action_angles = [0., 0., 0.]
        action_gripper = [1.0]
        wrist_angle = [0.0]

        if self.env._hp.action_mode == '3trans':
            action = np.concatenate((action_xyz, action_gripper))
        elif self.env._hp.action_mode == '3trans1rot':
            action = np.concatenate((action_xyz, wrist_angle, action_gripper))
        action = self.env.normalize_action(action)
        agent_info = dict()
        return action, agent_info