## This is the highest level interface to interact with the widowx setup.

import time
import cv2
import argparse
import numpy as np
from typing import Optional, Tuple, Any

# from: https://github.com/youliangtan/edgeml
from edgeml.interfaces import EdgeClient, EdgeServer, EdgeConfig

##############################################################################

DefaultEdgeConfig = EdgeConfig(
        port_number = 5566,
        action_keys = ["init", "move", "gripper", "reset"],
        observation_keys = ["image", "proprio"],
        broadcast_port= 5566 + 1,
    )

print_red = lambda x: print("\033[91m{}\033[00m".format(x))

##############################################################################

class WidowXEdgeServer():
    """
    This is the highest level abstraction of the widowx setup. We will run
    this as a server, and we can have multiple clients connect to it and
    reveives the observation and control the widowx robot.
    """
    def __init__(self, port: int = 5000, testing: bool = True):
        edgeml_config = DefaultEdgeConfig
        edgeml_config.port_number = port

        self.testing = testing
        self.bridge_env = None
        self.__server = EdgeServer(edgeml_config,
                                   obs_callback=self.__observe,
                                   act_callback=self.__action)

    def start(self, threaded: bool = False):
        """
        This starts the server. Default is blocking.
        """
        self.__server.start(threaded)

    def __action(self, type: str, req_payload: dict) -> dict:
        if type == "init":
            if self.testing:
                print_red("WARNING: Running in testing mode, \
                    no env will be initialized.")
                return {}

            from widowx_envs.widowx_env import BridgeDataRailRLPrivateWidowX
            from multicam_server.topic_utils import IMTopic
            from tf.transformations import quaternion_from_euler
            from tf.transformations import quaternion_matrix

            # brute force way to convert json to IMTopic
            env_params = None
            if req_payload:
                cam_imtopic = []
                for cam in req_payload["camera_topics"]:
                    imtopic_obj = IMTopic.model_validate_json(cam)
                    cam_imtopic.append(imtopic_obj)
                req_payload["camera_topics"] = cam_imtopic
                env_params = req_payload

            def get_tf_mat(pose):
                # convert pose to a 4x4 tf matrix, rpy to quat
                quat = quaternion_from_euler(pose[3], pose[4], pose[5])
                tf_mat = quaternion_matrix(quat)
                tf_mat[:3, 3] = pose[:3]
                return tf_mat
            
            self.get_tf_mat = get_tf_mat
            self.bridge_env = BridgeDataRailRLPrivateWidowX(
                env_params, fixed_image_size=128)
            print("Initialized bridge env.")

        elif type == "gripper":
            self.__gripper(req_payload["open"])
        elif type == "move":
            self.__move(req_payload["pose"], req_payload["duration"])
        elif type == "reset":
            self.__reset()
        return {}
    
    def __observe(self, types: list) -> dict:
        if self.bridge_env:
            # we will default return image and proprio only
            obs = self.bridge_env.current_obs()
            obs = {"image": obs["image"], "proprio": obs["proprio"]}
        else:
            # use dummy img with random noise
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            obs = {"image": img, "proprio": {}}
            print_red("WARNING: No bridge env not initialized.")
        return obs

    def __gripper(self, open: float):
        if self.bridge_env:
            if open > 0.5: # convert to bool, for future float support
                self.bridge_env.open_gripper()
            else:
                self.bridge_env.close_gripper()
        else:
            print_red("WARNING: No bridge env not initialized.")

    def __move(self, pose: np.ndarray, duration: float) -> bool:
        if self.bridge_env:
            # TODO: test this! Is better to use the controller
            # to move directly to the pose, instead of the gym api
            # self.bridge_env.step(pose)
            self.bridge_env.controller().move_to_eep(self.get_tf_mat(pose))
        else:
            print_red("WARNING: No bridge env not initialized.")

    def __reset(self):
        if self.bridge_env:
            self.bridge_env.reset()
            self.bridge_env.start()
        else:
            print_red("WARNING: No bridge env not initialized.")

    def stop(self):
        """Stop the server."""
        self.__server.stop()

##############################################################################

class WidowXClient():
    def __init__(self,
                 host: str = "localhost",
                 port: int = 5000,
                 env_params: dict = {}
                 ):
        edgeml_config = DefaultEdgeConfig
        edgeml_config.port_number = port
        self.__client = EdgeClient(host, edgeml_config)
        self.init(env_params)
        print("Initialized widowx client.")

    def move(self, pose: np.ndarray, duration: float) -> bool:
        """
        Command the arm to move to a given pose in space.
            :param pose: dim of 6, [x, y, z, roll, pitch, yaw]
        """
        assert len(pose) == 6
        self.__client.act("move", {"pose": pose, "duration": duration})

    def move_gripper(self, state: float):
        """Open or close the gripper. 1.0 is open, 0.0 is closed."""
        self.__client.act("gripper", {"open": state})
    
    def init(self, env_params: dict):
        """Initialize the environment."""
        self.__client.act("init", env_params)

    def reset(self):
        """Reset the arm to the neutral position."""
        self.__client.act("reset", {})

    def get_observation(self) -> Optional[Tuple[np.ndarray, dict]]:
        """
        Get the current camera image and proprioceptive state.
            :return Tuple of (image, proprio) or None if no observation is avail.
        """
        res = self.__client.obs()
        return res["image"], res["proprio"] if res else None
    
    def stop(self):
        """Stop the client."""
        self.__client.stop()

##############################################################################

if __name__ == "__main__":
    # NOTE: This is just for Testing
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', action='store_true')
    parser.add_argument('--client', action='store_true')
    parser.add_argument('--ip', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=5556)
    parser.add_argument('--show_video', action='store_true') # TODO
    args = parser.parse_args()

    if args.server:
        widowx_server = WidowXEdgeServer(port=args.port, testing=True)
        widowx_server.start()
        widowx_server.stop()

    if args.client:
        env_params = {
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
        widowx_client = WidowXClient(host=args.ip, port=args.port, env_params=env_params)

        # Testing
        time.sleep(1)
        widowx_client.move(np.array([0.1, 0.1, 0.1, 0, 0, 0]), 0.2)
        widowx_client.move_gripper(0.0)
        time.sleep(1)
        widowx_client.move_gripper(1.0)
        time.sleep(1)

        img, proprio = widowx_client.get_observation()
        print(proprio)
        cv2.imshow("img", img)
        cv2.waitKey(0)

        widowx_client.stop()
        print("Done.")
