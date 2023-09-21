## This is the highest level interface to interact with the widowx setup.

import time
import cv2
import argparse
import numpy as np
from typing import Optional, Tuple, Any
from widowx_envs.utils.exceptions import Environment_Exception

# install from: https://github.com/youliangtan/edgeml
from edgeml.interfaces import ActionClient, ActionServer, ActionConfig

##############################################################################

class WidowXConfigs:
    DefaultEnvParams = {
        "fix_zangle": 0.1,
        "move_duration": 0.2,
        "adaptive_wait": True,
        "move_to_rand_start_freq": 1,
        "override_workspace_boundaries": [
            [0.1, -0.15, -0.1, -1.57, 0],
            [0.45, 0.25, 0.18, 1.57, 0],
        ],
        "action_clipping": "xyz",
        "catch_environment_except": False,
        "start_state": None,
        "return_full_image": False,
        "camera_topics": [{"name": "/D435/color/image_raw", "flip": True}],
    }

    DefaultActionConfig = ActionConfig(
        port_number = 5556,
        action_keys = ["init", "move", "gripper", "reset", "step_action"],
        observation_keys = ["image", "proprio"],
        broadcast_port= 5556 + 1,
    )

print_red = lambda x: print("\033[91m{}\033[00m".format(x))

##############################################################################

class WidowXStatus:
    NO_CONNECTION = 0
    SUCCESS = 1
    EXECUTION_FAILURE = 2
    NOT_INITIALIZED = 3

##############################################################################

class WidowXActionServer():
    """
    This is the highest level abstraction of the widowx setup. We will run
    this as a server, and we can have multiple clients connect to it and
    reveives the observation and control the widowx robot.
    """
    def __init__(self, port: int = 5556, testing: bool = False):
        edgeml_config = WidowXConfigs.DefaultActionConfig
        edgeml_config.port_number = port
        edgeml_config.broadcast_port = port + 1

        self.testing = testing  # TODO: remove this soon
        self.bridge_env = None
        self.__server = ActionServer(edgeml_config,
                                   obs_callback=self.__observe,
                                   act_callback=self.__action)

    def start(self, threaded: bool = False):
        """
        This starts the server. Default is blocking.
        """
        self.__server.start(threaded)

    def __action(self, type: str, req_payload: dict) -> dict:
        return_status = WidowXStatus.SUCCESS
        if type == "init":
            if self.testing:
                print_red("WARNING: Running in testing mode, \
                    no env will be initialized.")
                return {"status": WidowXStatus.NOT_INITIALIZED}

            elif self.bridge_env and not req_payload["reinit"]:
                print_red("env already initialized")
                return {"status": WidowXStatus.SUCCESS}

            from widowx_envs.widowx_env import BridgeDataRailRLPrivateWidowX
            from multicam_server.topic_utils import IMTopic
            from tf.transformations import quaternion_from_euler
            from tf.transformations import quaternion_matrix

            # brute force way to convert json to IMTopic
            env_params = None
            _env_params = req_payload["env_params"]
            cam_imtopic = []
            for cam in _env_params["camera_topics"]:
                imtopic_obj = IMTopic.model_validate(cam)
                cam_imtopic.append(imtopic_obj)
            _env_params["camera_topics"] = cam_imtopic
            env_params = _env_params

            def get_tf_mat(pose):
                # convert pose to a 4x4 tf matrix, rpy to quat
                quat = quaternion_from_euler(pose[3], pose[4], pose[5])
                tf_mat = quaternion_matrix(quat)
                tf_mat[:3, 3] = pose[:3]
                return tf_mat

            self.get_tf_mat = get_tf_mat
            self.bridge_env = BridgeDataRailRLPrivateWidowX(
                env_params, fixed_image_size=req_payload["image_size"])
            print("Initialized bridge env.")

        elif type == "gripper":
            return_status = self.__gripper(req_payload["open"])
        elif type == "move":
            return_status = self.__move(
                req_payload["pose"], req_payload["duration"], req_payload["blocking"])
        elif type == "step_action":
            return_status = self.__step_action(req_payload["action"])
        elif type == "reset":
            return_status = self.__reset()
        return {"status": return_status}
    
    def __observe(self, types: list) -> dict:
        if self.bridge_env:
            # we will default return image and proprio only
            obs = self.bridge_env.current_obs()
            obs = {
                    "image": obs["image"],
                    "state": obs["state"],
                    "full_image": obs["full_image"][0]
                  }
        else:
            # use dummy img with random noise
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            obs = {"image": img, "state": {}, "full_image": img}
            print_red("WARNING: No bridge env not initialized.")
        return obs

    def __gripper(self, open: float) -> WidowXStatus:
        if self.bridge_env is None:
            print_red("WARNING: No bridge env not initialized.")
            return WidowXStatus.NOT_INITIALIZED
        
        if open > 0.5: # convert to bool, for future float support
            self.bridge_env.controller().open_gripper()
        else:
            self.bridge_env.controller().close_gripper()
        return WidowXStatus.SUCCESS

    def __move(self, pose: np.ndarray, duration: float, blocking: bool = True
               ) -> WidowXStatus:
        if self.bridge_env is None:
            print_red("WARNING: No bridge env not initialized.")
            return WidowXStatus.NOT_INITIALIZED

        if pose.shape == (4, 4):
            eep = pose
        else:
            eep = self.get_tf_mat(pose)
        try:
            self.bridge_env.controller().move_to_eep(
                eep, blocking=blocking, duration=duration)
            self.bridge_env._reset_previous_qpos()
        except Environment_Exception as e:
            print_red("Move execution error: {}".format(e))
            return WidowXStatus.EXECUTION_FAILURE
        return WidowXStatus.SUCCESS

    def __step_action(self, action: np.ndarray) -> WidowXStatus:
        if self.bridge_env is None:
            print_red("WARNING: No bridge env not initialized.")
            return WidowXStatus.NOT_INITIALIZED

        self.bridge_env.step(action)
        return WidowXStatus.SUCCESS

    def __reset(self) -> WidowXStatus:
        if self.bridge_env is None:
            print_red("WARNING: No bridge env not initialized.")
            return WidowXStatus.NOT_INITIALIZED

        # self.bridge_env.controller().move_to_neutral(duration=1.0)
        # self.bridge_env.controller().open_gripper()
        self.bridge_env.reset()
        self.bridge_env.start()
        return WidowXStatus.SUCCESS

    def stop(self):
        """Stop the server."""
        self.__server.stop()
        return WidowXStatus.SUCCESS

##############################################################################

class WidowXClient():
    def __init__(self,
                 host: str = "localhost",
                 port: int = 5556,
                 ):
        """
        Args:
            :param host: the host ip address
            :param port: the port number
        """
        edgeml_config = WidowXConfigs.DefaultActionConfig
        edgeml_config.port_number = port
        edgeml_config.broadcast_port = port + 1
        self.__client = ActionClient(host, edgeml_config)
        print("Initialized widowx client.")

    def init(self,
             env_params: dict,
             image_size: int = 256,
             reinit: bool=False
            ) -> WidowXStatus:
        """
        Initialize the environment.
            :param env_params: a dict of env params
            :param image_size: the size of the image to return
            :param reinit: whether to reinit the env, default False
        """
        payload = {"env_params": env_params,
                   "image_size": image_size,
                   "reinit": reinit}
        res = self.__client.act("init", payload)
        return WidowXStatus.NO_CONNECTION if res is None else res["status"]

    def move(self,
             pose: np.ndarray,
             duration: float = 1.0,
             blocking: bool = True,
            ) -> WidowXStatus:
        """
        Command the arm to move to a given pose in space.
            :param pose: dim of 6, [x, y, z, roll, pitch, yaw] or
                         a 4x4 tf matrix
            :param duration: time to move to the pose. Not implemented
            :param blocking: whether to block until the move is done
        """
        assert len(pose) == 6 or pose.shape == (4, 4), "invalid pose shape"
        _payload = {"pose": pose, "duration": duration, "blocking": blocking}
        res = self.__client.act("move", _payload)
        return WidowXStatus.NO_CONNECTION if res is None else res["status"]

    def move_gripper(self, state: float) -> WidowXStatus:
        """Open or close the gripper. 1.0 is open, 0.0 is closed."""
        res = self.__client.act("gripper", {"open": state})
        return WidowXStatus.NO_CONNECTION if res is None else res["status"]

    def step_action(self, action: np.ndarray) -> WidowXStatus:
        """
        Step the action. size of 5 (3trans) or 7 (3trans1rot)
        Note that the action is in relative space.
        """
        assert len(action) in [5, 7], "invalid action shape"
        res = self.__client.act("step_action", {"action": action})
        return WidowXStatus.NO_CONNECTION if res is None else res["status"]

    def reset(self) -> WidowXStatus:
        """Reset the arm to the neutral position."""
        res = self.__client.act("reset", {})
        return WidowXStatus.NO_CONNECTION if res is None else res["status"]

    def get_observation(self) -> Optional[dict]:
        """
        Get the current camera image and proprioceptive state.
            :return a dict of observations
        """
        res = self.__client.obs()
        return res if res else None

    def stop(self):
        """Stop the client."""
        self.__client.stop()

##############################################################################

def show_video(client, duration, full_image=True):
    """This shows the video from the camera for a given duration."""
    start = time.time()
    while (time.time() - start) < duration:
        res = client.get_observation()
        if res is None:
            print("No observation available... waiting")
            continue

        if full_image:
            img = res["full_image"]
        else:
            img = res["image"]       
            # if img.shape[0] != 3:  # sanity check to make sure it's not flattened
            img = (img.reshape(3, 256, 256).transpose(1, 2, 0) * 255).astype(np.uint8)
        cv2.imshow("img", img)
        cv2.waitKey(100)  # 100 ms

def main():
    # NOTE: This is just for Testing
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', action='store_true')
    parser.add_argument('--client', action='store_true')
    parser.add_argument('--ip', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=5556)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    if args.server:
        widowx_server = WidowXActionServer(port=args.port, testing=args.test)
        widowx_server.start()
        widowx_server.stop()

    if args.client:
        widowx_client = WidowXClient(host=args.ip, port=args.port)

        # NOTE: this normally takes 10 seconds when first time init
        widowx_client.init(WidowXConfigs.DefaultEnvParams, image_size=256)

        # This ensures that the robot is ready to be controlled
        obs = None
        while obs is None:
            obs = widowx_client.get_observation()
            time.sleep(1)
            print("Waiting for robot to be ready...")

        # Coordinate Convention:
        #  - x: forward
        #  - y: left
        #  - z: up

        # move left up
        res = widowx_client.move(np.array([0.2, 0.1, 0.3, 0, 1.57, 1.57]), 0.2, blocking=False)
        assert args.test or res == WidowXStatus.SUCCESS, "move failed"
        show_video(widowx_client, 1.5)

        # close gripper
        print("Closing gripper...")
        res = widowx_client.move_gripper(0.0)
        assert args.test or res == WidowXStatus.SUCCESS, "gripper failed"
        show_video(widowx_client, 2.5)

        # move right down
        res = widowx_client.move(np.array([0.2, -0.1, 0.1, 0, 1.57, 0]), 0.2, blocking=False)
        assert args.test or res == WidowXStatus.SUCCESS, "move failed"
        show_video(widowx_client, 1.5)

        # open gripper
        print("Opening gripper...")
        res = widowx_client.move_gripper(1.0)
        assert args.test or res == WidowXStatus.SUCCESS, "gripper failed"
        show_video(widowx_client, 2.5)

        widowx_client.stop()
        print("Done all")

if __name__ == '__main__':
    main()
