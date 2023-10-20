import numpy as np

from widowx_envs.widowx.widowx_env import StateReachingWidowX, \
    ImageReachingWidowX
from widowx_envs.utils.multicam_server_rospkg.src.topic_utils import IMTopic


class GraspWidowXEnv(ImageReachingWidowX):
    def __init__(self, env_params=None, publish_images=True, fixed_image_size=64):
        from gym import spaces

        super(GraspWidowXEnv, self).__init__(
            env_params=env_params, publish_images=publish_images,
            fixed_image_size=fixed_image_size,
        )

        delta_limit = 0.05
        # max step size: 5cm for each axis
        # for gripper action, 0 is fully closed, 1 is fully open,
        # continuous interpolation in the middle

        self.action_space = spaces.Box(
            np.asarray([-0.05, -0.05, -0.05, -0.25, -0.25, -0.25, 0.]),
            np.asarray([0.05, 0.05, 0.05, 0.25, 0.25, 0.25, 1.0]),
            dtype=np.float32)
        self.startpos = np.asarray([.1167643, -0.0190529, 0.24011597]) #np.asarray([0., -0.25, 0.15])

    def step(self, action):
        # first three dimensions are delta x, y, z
        # Last action is gripper action
        # for gripper action, 0 is fully closed, 1 is fully open,
        # continuous interpolation in the middle

        action = np.clip(action, self.action_space.low, self.action_space.high)
        obs = super(StateReachingWidowX, self).step(action)
        reward = 0.
        done = False

        # if obs['state'][-1] > 0.7:
        #     self.is_gripper_open = True
        # else:
        #     self.is_gripper_open = False
        # obs = self._get_obs()

        if self.publish_images:
            self._publish_image(obs['image'])
        return obs, reward, done, {}

    def reset(self, itraj=None, reset_state=None):
        """
        Resets the environment and returns initial observation
        :return: obs dict (look at step(self, action) for documentation)
        """
        self._controller.open_gripper(True)
        # self.is_gripper_open = True

        if not self._hp.skip_move_to_neutral:
            self._controller.move_to_neutral(duration=1.5)

        # if itraj % self._hp.move_to_rand_start_freq == 0:
        #     self.move_to_startstate()
        zangle = 0.
        self._controller.move_to_state(self.startpos, zangle, duration=1.5)
        self._reset_previous_qpos()

        # time.sleep(self._hp.wait_time)  # sleep is already called in self._reset_previous_qpos()
        return self._get_obs()

    def _get_obs(self):
        full_obs = super(StateReachingWidowX, self)._get_obs()
        image = full_obs['images'][0]
        ee_coord = full_obs['full_state'][:3]
        processed_image = self._get_processed_image(image)

        obs = {'image': processed_image, 'state': self.get_full_state(),
               'joints': full_obs['qpos'], 'ee_coord': ee_coord}

        if self._hp.return_full_image:
            obs['full_image'] = image
        # obs['gripper'] = full_obs['state'][-1]  # this dimension is not being updated for now
        return obs

    def _default_hparams(self):
        default_dict = {
            'camera_topics': [IMTopic('/cam0/image_raw')],
            'override_workspace_boundaries': [[0.19, -0.08, 0.029, -1.57, 0],
                                              [0.31, 0.08, 0.15,  1.57, 0]]
        }
        parent_params = super(GraspWidowXEnv, self)._default_hparams()
        parent_params.update(default_dict)
        return parent_params


if __name__ == '__main__':
    # pass
    env = GraspWidowXEnv({'workspace_rotation_angle_z': -1.57,
                              'transpose_image_to_chw': True})
    env.reset()
    # import IPython; IPython.embed()

